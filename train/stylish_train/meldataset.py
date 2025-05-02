# coding: utf-8
import os.path as osp
import numpy as np
import soundfile as sf
import librosa
from librosa.filters import mel as librosa_mel_fn
import tqdm

import torch
import torchaudio
import torch.utils.data
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()

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


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_list,
        root_path,
        text_cleaner,
        model_config,
        pitch_path,
        alignment_path,
    ):
        self.pitch = {}
        with safe_open(pitch_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.pitch[key] = f.get_tensor(key)
        self.alignment = {}
        if osp.isfile(alignment_path):
            with safe_open(alignment_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.alignment[key] = f.get_tensor(key)
        self.data_list = []
        sentences = []
        for line in data_list:
            fields = line.strip().split("|")
            if len(fields) == 3:
                fields.append(" ")
            if len(fields) != 4:
                exit("Dataset lines must have 4 |-delimited fields: " + fields)
            self.data_list.append(fields)
            sentences.append(fields[3])
        self.sentences = sentences
        self.text_cleaner = text_cleaner

        self.df = pd.DataFrame(self.data_list)
        self.model_config = model_config

        self.to_align_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80,  # align seems to perform worse on higher n_mels
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length,
            # sample_rate=model_config.sample_rate,
        )

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=model_config.n_mels,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length,
            # sample_rate=model_config.sample_rate,
        )

        # self.min_length = min_length
        # with open(
        #     hf_hub_download(
        #         repo_id="stylish-tts/train-ood-texts",
        #         repo_type="dataset",
        #         filename="OOD_texts.txt",
        #     ),
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     tl = f.readlines()
        # idx = 1 if ".wav" in tl[0].split("|")[0] else 0
        # self.ptexts = [t.split("|")[idx] for t in tl]

        self.root_path = root_path
        self.multispeaker = model_config.multispeaker
        self.sample_rate = model_config.sample_rate
        self.hop_length = model_config.hop_length

    # def to_mel(self, wave_tensor):
    #     return mel_spectrogram(
    #         y=wave_tensor,
    #         n_fft=self.model_config.n_fft,
    #         num_mels=self.model_config.n_mels,
    #         sampling_rate=self.model_config.sample_rate,
    #         hop_size=self.model_config.hop_length,
    #         win_size=self.model_config.win_length,
    #         fmin=50,
    #         fmax=550,
    #     )

    def preprocess(self, wave, align=False):
        mean, std = -4, 4
        # wave_tensor = torch.from_numpy(wave).float()
        wave_tensor = wave
        if align:
            mel_tensor = self.to_align_mel(wave_tensor)
        else:
            mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def time_bins(self):
        sample_lengths = []
        iterator = tqdm.tqdm(
            iterable=self.data_list,
            desc="Calculating segment lengths",
            total=len(self.data_list),
            unit="segments",
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {remaining} ",
            initial=0,
            colour="MAGENTA",
            dynamic_ncols=True,
        )
        for data in iterator:
            wave_path = data[0]
            wave, sr = sf.read(osp.join(self.root_path, wave_path))
            wave_len = wave.shape[0]
            if sr != self.sample_rate:
                wave_len *= self.sample_rate / sr
            sample_lengths.append(wave_len)
        iterator.clear()
        iterator.close()
        time_bins = {}
        for i in range(len(sample_lengths)):
            phonemes = self.data_list[i][1]
            bin_num = get_time_bin(sample_lengths[i], self.hop_length)
            if get_frame_count(bin_num) < len(phonemes):
                exit(
                    f"Segment audio is too short for the number of phonemes. Remove it from the dataset: {self.data_list[i][0]}"
                )
            if len(phonemes) < 1:
                exit(
                    f"Segment does not have any phonmes. Remove it from the dataset: {self.data_list[i][0]}"
                )
            if len(phonemes) > 510:
                exit(
                    f"Segment has too many phonemes. Segments must be of size <= 510. Remove it from the dataset: {self.data_list[i][0]}"
                )
            if bin_num != -1:
                if bin_num not in time_bins:
                    time_bins[bin_num] = []
                time_bins[bin_num].append(i)
            # else:
            #     exit(
            #         f"Segment Length Too Short. Must be at least 0.25 seconds: {self.data_list[i][0]}"
            #     )
        return time_bins

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id, mel_tensor, voiced_tensor, align_mel = (
            self._load_tensor(data)
        )

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]
        align_mel = align_mel[:, : (length_feature - length_feature % 2)]

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

        # ps = ""
        ref_text = torch.LongTensor()
        # while len(ps) < self.min_length:
        #     rand_idx = np.random.randint(0, len(self.ptexts) - 1)
        #     ps = self.ptexts[rand_idx]

        #     text = self.text_cleaner(ps)
        #     text.insert(0, 0)
        #     text.append(0)

        #     ref_text = torch.LongTensor(text)

        pitch = None
        if path in self.pitch:
            pitch = torch.nan_to_num(self.pitch[path].detach().clone())
        alignment = None
        if path in self.alignment:
            alignment = self.alignment[path]
            # alignment = torch.nn.functional.interpolate(alignment, scale_factor=2, mode="nearest")
            alignment = alignment.detach()
        sentence_embedding = torch.from_numpy(
            sbert.encode([self.sentences[idx]], show_progress_bar=False)
        ).float()

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
            sentence_embedding,
            voiced_tensor,
            align_mel,
            alignment,
        )

    def _load_tensor(self, data):
        wave_path, text, speaker_id, _ = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != self.sample_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sample_rate)
            logger.debug(f"{wave_path}, {sr}")

        pad_start = 5000
        pad_end = 5000
        time_bin = get_time_bin(wave.shape[0], self.hop_length)
        if time_bin != -1:
            frame_count = get_frame_count(time_bin)
            pad_start = (frame_count * self.hop_length - wave.shape[0]) // 2
            pad_end = frame_count * self.hop_length - wave.shape[0] - pad_start
        wave = np.concatenate(
            [np.zeros([pad_start]), wave, np.zeros([pad_end])], axis=0
        )
        wave = torch.from_numpy(wave).float()

        text = self.text_cleaner(text)
        voiced = self.text_cleaner.is_voiced(text)

        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)

        voiced.insert(0, 0)
        voiced.append(0)
        voiced = torch.tensor(voiced, dtype=torch.float32)

        mel_tensor = self.preprocess(wave, align=False).squeeze()
        align_mel = self.preprocess(wave, align=True).squeeze()

        return (wave, text, speaker_id, mel_tensor, voiced, align_mel)

    def _load_data(self, data):
        max_mel_length = 192
        wave, text_tensor, speaker_id, mel_tensor = self._load_tensor(data)

        mel_length = mel_tensor.size(1)
        if mel_length > max_mel_length:
            random_start = np.random.randint(0, mel_length - max_mel_length)
            mel_tensor = mel_tensor[:, random_start : random_start + max_mel_length]

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
        sentence_embeddings = torch.zeros(batch_size, 384).float()
        voiced = torch.zeros((batch_size, max_text_length)).float()
        align_mels = torch.zeros((batch_size, 80, max_mel_length)).float()
        alignments = torch.zeros((batch_size, max_text_length, max_mel_length // 2))

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
            sentence,
            voiced_one,
            align_mel,
            alignment,
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
            sentence_embeddings[bid] = sentence
            voiced[bid, :text_size] = voiced_one
            align_mels[bid, :, :mel_size] = align_mel
            alignments[bid, :text_size, : mel_size // 2] = alignment

        result = (
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
            sentence_embeddings,
            voiced,
            align_mels,
            alignments,
        )
        return result


def build_dataloader(
    dataset,
    time_bins,
    validation=False,
    num_workers=1,
    device="cpu",
    collate_config={},
    probe_bin=None,
    probe_batch_size=None,
    drop_last=True,
    multispeaker=False,
    epoch=1,
    *,
    train,
):
    collate_config["multispeaker"] = multispeaker
    collate_fn = Collater(**collate_config)
    drop_last = not validation and probe_batch_size is not None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=DynamicBatchSampler(
            time_bins,
            shuffle=(not validation),
            drop_last=drop_last,
            force_bin=probe_bin,
            force_batch_size=probe_batch_size,
            epoch=epoch,
            train=train,
        ),
        collate_fn=collate_fn,
        pin_memory=False,  # (device != "cpu"),
    )

    return data_loader


class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        time_bins,
        shuffle=True,
        seed=0,
        drop_last=False,
        epoch=1,
        force_bin=None,
        force_batch_size=None,
        *,
        train,
    ):
        self.time_bins = time_bins
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
        self.train = train

    def __iter__(self):
        # provided_steps = 0
        samples = {}
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.force_bin is not None:
            samples = {self.force_bin: self.time_bins[self.force_bin]}
        else:
            for key in self.time_bins.keys():
                if self.get_batch_size(key) <= 0:
                    continue
                if not self.drop_last or len(
                    self.time_bins[key] >= self.get_batch_size(key)
                ):
                    if self.shuffle:
                        order = torch.randperm(len(self.time_bins[key]), generator=g)
                        current = []
                        for index in order:
                            current.append(self.time_bins[key][index])
                        samples[key] = current
                    else:
                        samples[key] = self.time_bins[key]

        sample_keys = list(samples.keys())
        while len(sample_keys) > 0:
            if self.shuffle:
                index = torch.randint(0, len(sample_keys), [1], generator=g)[0]
            else:
                index = 0
            key = sample_keys[index]
            current_samples = samples[key]
            batch_size = min(len(current_samples), self.get_batch_size(key))
            done = False
            while not done:
                batch = current_samples[:batch_size]
                remaining = current_samples[batch_size:]
                if len(remaining) == 0 or (
                    self.drop_last and len(remaining) < batch_size
                ):
                    del samples[key]
                    done = True
                else:
                    samples[key] = remaining
                    current_samples = remaining
                yield batch
                self.train.stage.load_batch_sizes()
            sample_keys = list(samples.keys())

    def __len__(self):
        return self.train.stage.get_steps_per_epoch()
        total = 0
        for key in self.time_bins.keys():
            val = self.time_bins[key]
            total_batch = self.train.stage.get_batch_size(key)
            if total_batch > 0:
                total += len(val) // total_batch
                if not self.drop_last and len(val) % total_batch != 0:
                    total += 1
        return total

    def set_epoch(self, epoch):
        self.epoch = epoch

    def probe_batch(self, new_bin, batch_size):
        self.force_bin = new_bin
        if len(self.time_bins[new_bin]) < batch_size:
            batch_size = len(self.time_bins[new_bin])
        self.force_batch_size = batch_size
        return batch_size

    def get_batch_size(self, key):
        if self.force_batch_size is not None:
            return self.force_batch_size
        else:
            return self.train.stage.get_batch_size(key)


def get_frame_count(i):
    return i * 20 + 20 + 40


def get_time_bin(sample_count, hop_length):
    result = -1
    frames = sample_count // hop_length
    if frames >= 20:
        result = (frames - 20) // 20
    return result


def get_padded_time_bin(sample_count, hop_length):
    frames = sample_count // hop_length
    return (frames - 60) // 20
