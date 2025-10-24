import random
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

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        duration_processor,
    ):
        self.pitch = {}
        with safe_open(pitch_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.pitch[key] = f.get_tensor(key)
        durations = torch.zeros([16], device="cpu")
        self.alignment = {}
        if osp.isfile(alignment_path):
            with safe_open(alignment_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    alignment = f.get_tensor(key)
                    self.alignment[key] = alignment
                    dur = (
                        duration_processor.dur_to_class(alignment[0])
                        .long()
                        .cpu()
                        .numpy()
                    )
                    dur = torch.Tensor(np.bincount(dur, minlength=16))
                    durations += dur
        self.duration_weights = durations.sum() / (durations * durations.shape[0])
        self.data_list = []
        sentences = []
        for line in data_list:
            fields = line.strip().split("|")
            if len(fields) != 4:
                exit("Dataset lines must have 4 |-delimited fields: " + fields)
            self.data_list.append(fields)
            sentences.append(fields[3])
        self.sentences = sentences
        self.text_cleaner = text_cleaner

        self.model_config = model_config
        self.root_path = root_path
        self.multispeaker = model_config.multispeaker
        self.sample_rate = model_config.sample_rate
        self.hop_length = model_config.hop_length

    def time_bins(self):
        sample_lengths = []
        total_audio_length = 0
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
            info = sf.info(osp.join(self.root_path, wave_path))
            frames = info.frames
            if info.samplerate != self.sample_rate:
                frames = int(info.frames * self.sample_rate / info.samplerate)
            sample_lengths.append(frames)
            total_audio_length += frames / info.samplerate
        iterator.clear()
        iterator.close()
        time_bins = {}
        time_per_bin = {}
        logger.info(f"Total segment lengths: {total_audio_length / 3600.0:.2f}h")
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
                if bin_num not in time_per_bin:
                    time_per_bin[bin_num] = 0
                time_per_bin[bin_num] += sample_lengths[i] / self.sample_rate
            else:
                exit(
                    f"Segment Length Too Short. Must be at least 0.25 seconds: {self.data_list[i][0]}"
                )
        return time_bins, time_per_bin

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id = self._load_tensor(data)

        pitch = None
        if path in self.pitch:
            pitch = torch.nan_to_num(self.pitch[path].detach().clone())
        alignment = None
        if path in self.alignment:
            alignment = self.alignment[path]
            alignment = alignment.detach()
        else:
            # TODO need to warn here or check for alignment stage and skip
            alignment = torch.zeros(
                (3, text_tensor.shape[0]),
                dtype=torch.float32,  # Match Collater's target dtype
            )

        return (
            speaker_id,
            text_tensor,
            path,
            wave,
            pitch,
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
            # pad_start = (frame_count * self.hop_length - wave.shape[0]) // 2
            # pad_end = frame_count * self.hop_length - wave.shape[0] - pad_start
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

        return (wave, text, speaker_id)


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False, multispeaker=False, *, stage, hop_length):
        self.return_wave = return_wave
        self.multispeaker = multispeaker
        self.stage = stage
        self.hop_length = hop_length

    def __call__(self, batch):
        batch_size = len(batch)

        max_text_length = max([b[1].shape[0] for b in batch])
        mel_length = batch[0][3].shape[-1] // self.hop_length

        speaker_out = torch.zeros((batch_size)).long()
        texts = torch.zeros((batch_size, max_text_length)).long()
        text_lengths = torch.zeros(batch_size).long()
        paths = ["" for _ in range(batch_size)]
        waves = torch.zeros((batch_size, batch[0][3].shape[-1])).float()
        pitches = torch.zeros((batch_size, mel_length)).float()
        alignments = torch.zeros((batch_size, 1, max_text_length))
        # alignments = torch.zeros((batch_size, max_text_length, mel_length))
        # alignments = torch.zeros((batch_size, max_text_length, mel_length // 2))

        for bid, (
            speaker,
            text,
            path,
            wave,
            pitch,
            duration,
        ) in enumerate(batch):
            speaker_out[bid] = speaker

            text_size = text.size(0)
            texts[bid, :text_size] = text

            text_lengths[bid] = text_size
            paths[bid] = path
            waves[bid] = wave
            # TODO: The 4 here and in alignments is a kludge, probably should be removed
            if self.stage != "alignment":
                if pitch is None:
                    exit(f"Pitch not found for segment {path}")
                pitches[bid] = torch.repeat_interleave(pitch, 4, dim=-1)

            # alignments[bid, :text_size, : mel_length // 2] = duration
            # pred_dur = duration[0]
            # for i in range(pred_dur.shape[0] - 1):
            #     if pred_dur[i] > 1 and pred_dur[i + 1] > 1:
            #         pick = random.random()
            #         if pick < duration[1][i]:
            #             pred_dur[i] += 1
            #             pred_dur[i + 1] -= 1
            #         elif pick < duration[1][i] + duration[2][i]:
            #             pred_dur[i] -= 1
            #             pred_dur[i + 1] += 1
            # alignment = self.train.duration_processor.duration_to_alignment(pred_dur)
            # if self.stage != "alignment":
            #     if alignment.shape[1] != mel_length:
            #         exit(f"Alignment for segment {path} did not match audio length")
            #     alignments[bid, :text_size, :mel_length] = alignment
            alignments[bid, :1, :text_size] = duration[:1] * 4

        result = (
            waves,
            texts,
            text_lengths,
            paths,
            pitches,
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
    stage,
    hop_length,
    train,
):
    collate_config["multispeaker"] = multispeaker
    collate_fn = Collater(stage=stage, hop_length=hop_length, **collate_config)
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
        pin_memory=False,
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
            index = 0
            if self.shuffle:
                total_weight = 0
                for key in sample_keys:
                    total_weight += len(samples[key]) // self.get_batch_size(key) + 1
                weight = torch.randint(0, total_weight, [1], generator=g)[0]
                for i in range(len(sample_keys)):
                    weight -= (
                        len(samples[sample_keys[i]])
                        // self.get_batch_size(sample_keys[i])
                        + 1
                    )
                    if weight <= 0:
                        index = i
                        break
            key = sample_keys[index]
            current_samples = samples[key]
            batch_size = min(len(current_samples), self.get_batch_size(key))
            batch = current_samples[:batch_size]
            remaining = current_samples[batch_size:]
            if len(remaining) == 0 or (self.drop_last and len(remaining) < batch_size):
                del samples[key]
            else:
                samples[key] = remaining
            yield batch
            if self.train is not None:
                self.train.stage.load_batch_sizes()
            sample_keys = list(samples.keys())

    def __len__(self):
        if self.train is not None:
            return self.train.stage.get_steps_per_epoch()
        else:
            result = 0
            for key in self.time_bins.keys():
                result += len(self.time_bins[key])
            return result
        # total = 0
        # for key in self.time_bins.keys():
        #     val = self.time_bins[key]
        #     total_batch = self.train.stage.get_batch_size(key)
        #     if total_batch > 0:
        #         total += len(val) // total_batch
        #         if not self.drop_last and len(val) % total_batch != 0:
        #             total += 1
        # return total

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
        elif self.train is not None:
            return self.train.stage.get_batch_size(key)
        else:
            return 1


def get_frame_count(i):
    return i * 20 + 20 + 40


# TODO: More badness. Make it hop_length again
def get_time_bin(sample_count, hop_length):
    result = -1
    frames = sample_count // 300  # hop_length
    if frames >= 20:
        result = (frames - 20) // 20
    return result


def get_padded_time_bin(sample_count, hop_length):
    frames = sample_count // 300  # hop_length
    return (frames - 60) // 20
