# coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa
import gc
import json

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torch.utils.data
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from librosa.filters import mel as librosa_mel_fn

import logging
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

mel_window = {}


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=True,
    in_dataset=False,
):
    global mel_window
    # device = torch.device("cpu") if in_dataset else y.device
    device = "cpu"
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in mel_window:
        mel_basis, hann_window = mel_window[ps]
        # print(mel_basis, hann_window)
        # mel_basis, hann_window = mel_basis.to(y.device), hann_window.to(y.device)
    else:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        mel_window[ps] = (mel_basis.clone(), hann_window.clone())

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.to(device),
        center=True,
        return_complex=True,
    )

    spec = mel_basis.to(device) @ spec.abs()
    # spec = spectral_normalize_torch(spec)

    return spec  # [batch_size,n_fft/2+1,frames]


# to_mel = torchaudio.transforms.MelSpectrogram(
#    n_mels=80, n_fft=2048, win_length=1200, hop_length=300, sample_rate=24000
# )
mean, std = -4, 4


def preprocess(wave):
    # wave_tensor = torch.from_numpy(wave).float()
    wave_tensor = wave
    # mel_tensor = to_mel(wave_tensor)
    mel_tensor = mel_spectrogram(
        y=wave_tensor,
        n_fft=2048,
        num_mels=80,
        sampling_rate=24000,
        hop_size=300,
        win_size=1200,
        fmin=50,
        fmax=8000,
    )
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def amp_pha_specturm(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )  # [batch_size, n_fft//2+1, frames, 2]

    log_amplitude = torch.log(
        stft_spec.abs() + 1e-5
    )  # [batch_size, n_fft//2+1, frames]
    phase = stft_spec.angle()  # [batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec.real, stft_spec.imag


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        root_path,
        sr=24000,
        data_augmentation=False,
        validation=False,
        OOD_data="Data/OOD_texts.txt",
        min_length=50,
        multispeaker=False,
        text_cleaner=None,
        pitch_path="",
    ):

        self.cache = {}
        self.pitch = {}
        with safe_open(pitch_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.pitch[key] = f.get_tensor(key)
        _data_list = [l.strip().split("|") for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = text_cleaner
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(
            hf_hub_download(
                repo_id="stylish-tts/train-ood-texts",
                repo_type="dataset",
                filename="OOD_texts.txt",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            tl = f.readlines()
        idx = 1 if ".wav" in tl[0].split("|")[0] else 0
        self.ptexts = [t.split("|")[idx] for t in tl]

        self.root_path = root_path
        self.multispeaker = multispeaker

    def time_bins(self):
        logger.info("Calculating sample lengths")
        sample_lengths = []
        for data in self.data_list:
            wave_path = data[0]
            wave, sr = sf.read(osp.join(self.root_path, wave_path))
            wave_len = wave.shape[0]
            if sr != 24000:
                wave_len *= 24000 / sr
            sample_lengths.append(wave_len)
        time_bins = {}
        for i in range(len(sample_lengths)):
            bin_num = get_time_bin(sample_lengths[i])
            if bin_num != -1:
                if bin_num not in time_bins:
                    time_bins[bin_num] = []
                time_bins[bin_num].append(i)
        logger.info("Finished sample lengths")
        return time_bins

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id, mel_tensor = self._cache_tensor(data)

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # get reference sample
        if self.multispeaker:
            ref_data = (
                (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            )
            ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        else:
            ref_data = []
            ref_mel_tensor, ref_label = None, ""

        # get OOD text

        ps = ""

        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]

            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)

        pitch = None
        if path in self.pitch:
            pitch = torch.nan_to_num(self.pitch[path].detach().clone())
        return (
            speaker_id,
            acoustic_feature,
            text_tensor,
            ref_text,
            ref_mel_tensor,
            ref_label,
            path,
            wave,
            pitch,
        )

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            logger.debug(f"{wave_path}, {sr}")

        pad_start = 5000
        pad_end = 5000
        time_bin = get_time_bin(wave.shape[0])
        if time_bin != -1:
            frame_count = get_frame_count(time_bin)
            pad_start = (frame_count * 300 - wave.shape[0]) // 2
            pad_end = frame_count * 300 - wave.shape[0] - pad_start
        wave = np.concatenate(
            [np.zeros([pad_start]), wave, np.zeros([pad_end])], axis=0
        )
        wave = torch.from_numpy(wave).float()

        text = self.text_cleaner(text)

        text.insert(0, 0)
        text.append(0)

        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _cache_tensor(self, data):
        path = data[0]
        # if path in self.cache:
        # (wave, text_tensor, speaker_id, mel_tensor) = self.cache[path]
        # else:
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()
        # self.cache[path] = (wave, text_tensor, speaker_id,
        #                    mel_tensor)
        return (wave, text_tensor, speaker_id, mel_tensor)

    def _load_data(self, data):
        wave, text_tensor, speaker_id, mel_tensor = self._cache_tensor(data)

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False, multispeaker=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        self.multispeaker = multispeaker

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ["" for _ in range(batch_size)]
        waves = torch.zeros(
            (batch_size, batch[0][7].shape[-1])
        ).float()  # [None for _ in range(batch_size)]
        pitches = torch.zeros((batch_size, max_mel_length)).float()
        log_amplitudes = torch.zeros(batch_size, 1025, lengths[0] + 1).float()
        phases = torch.zeros(batch_size, 1025, lengths[0] + 1).float()
        reals = torch.zeros(batch_size, 1025, lengths[0] + 1).float()
        imags = torch.zeros(batch_size, 1025, lengths[0] + 1).float()

        for bid, (
            label,
            mel,
            text,
            ref_text,
            ref_mel,
            ref_label,
            path,
            wave,
            pitch,
        ) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            if self.multispeaker:
                ref_mel_size = ref_mel.size(1)
                ref_mels[bid, :, :ref_mel_size] = ref_mel
                ref_labels[bid] = ref_label
            waves[bid] = wave
            if pitch is not None:
                pitches[bid] = pitch
            # TODO: hard coded fix
            log_amplitude, phase, rea, imag = amp_pha_specturm(
                wave, n_fft=2048, hop_size=300, win_size=1200
            )
            log_amplitudes[bid] = log_amplitude
            phases[bid] = phase
            reals[bid] = rea
            imags[bid] = imag

        return (
            waves,
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            output_lengths,
            ref_mels,
            paths,
            pitches,
            log_amplitudes,
            phases,
            reals,
            imags,
        )


def build_dataloader(
    dataset,
    time_bins,
    validation=False,
    batch_size={},
    num_workers=1,
    device="cpu",
    collate_config={},
    probe_bin=None,
    probe_batch_size=None,
    drop_last=True,
    multispeaker=False,
    epoch=1,
):

    collate_config["multispeaker"] = multispeaker
    collate_fn = Collater(**collate_config)
    drop_last = not validation and probe_batch_size is not None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=DynamicBatchSampler(
            time_bins,
            batch_size,
            shuffle=(not validation),
            drop_last=drop_last,
            num_replicas=1,
            rank=0,
            force_bin=probe_bin,
            force_batch_size=probe_batch_size,
            epoch=epoch,
        ),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader


class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        time_bins,
        batch_sizes,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        epoch=1,
        force_bin=None,
        force_batch_size=None,
    ):
        self.time_bins = time_bins
        self.batch_sizes = batch_sizes
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.epoch = epoch
        self.total_len = 0
        self.last_bin = None

        self.force_bin = force_bin
        self.force_batch_size = force_batch_size
        if force_bin is not None and force_batch_size is not None:
            self.drop_last = False

        total = 0
        for key in self.time_bins.keys():
            total += len(self.time_bins[key])
        for key in self.time_bins.keys():
            val = self.time_bins[key]
            total_batch = self.get_batch_size(key) * num_replicas
            if total_batch > 0:
                self.total_len += len(val) // total_batch
                if not self.drop_last and len(val) % total_batch != 0:
                    self.total_len += 1

    def __iter__(self):
        sampler_order = list(self.time_bins.keys())
        sampler_indices = []
        if self.force_bin is not None:
            sampler_order = [self.force_bin]
            sampler_indices = [0]
        elif self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            sampler_indices = torch.randperm(len(sampler_order), generator=g).tolist()
        else:
            sampler_indices = list(range(len(sampler_order)))

        for index in sampler_indices:
            key = sampler_order[index]
            if self.get_batch_size(key) <= 0:
                continue
            current_bin = self.time_bins[key]
            dist = torch.utils.data.distributed.DistributedSampler(
                current_bin,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            dist.set_epoch(self.epoch)
            sampler = torch.utils.data.sampler.BatchSampler(
                dist, self.get_batch_size(key), self.drop_last
            )
            # logger.debug(f"{key}:{self.get_batch_size(key)}")
            for item_list in sampler:
                self.last_bin = key
                yield [current_bin[i] for i in item_list]

    def __len__(self):
        return self.total_len

    def set_epoch(self, epoch):
        self.epoch = epoch

    def probe_batch(self, new_bin, batch_size):
        self.force_bin = new_bin
        if len(self.time_bins[new_bin]) < batch_size:
            batch_size = len(self.time_bins[new_bin])
        self.force_batch_size = batch_size
        return batch_size

    def get_batch_size(self, key):
        result = 1
        if self.force_batch_size is not None:
            result = self.force_batch_size
        elif str(key) in self.batch_sizes:
            result = self.batch_sizes[str(key)]
        return result


def get_frame_count(i):
    return i * 20 + 20 + 40


def get_time_bin(sample_count):
    result = -1
    frames = sample_count // 300
    if frames >= 20:
        result = (frames - 20) // 20
    return result
