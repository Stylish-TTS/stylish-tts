import click
import logging
import math
from os import path as osp
from pathlib import Path
import sys

from einops import rearrange
import numpy
from safetensors.torch import load_file, save_file
import soundfile
import torch
from torch.nn import functional as F
import torchaudio
import tqdm

from stylish_tts.train.models.text_aligner import tdnn_blstm_ctc_model_base
from stylish_tts.lib.config_loader import load_config_yaml, load_model_config_yaml
from stylish_tts.lib.text_utils import TextCleaner
from stylish_tts.train.utils import get_data_path_list, maximum_path
from stylish_tts.train.dataloader import (
    get_frame_count,
    get_time_bin,
    DynamicBatchSampler,
    FilePathDataset,
    Collater,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from stylish_tts.train.utils import calculate_mel
from stylish_tts.train.train_context import TrainContext
from stylish_tts.train.batch_manager import BatchManager
from stylish_tts.train.stage import prepare_batch
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_steps(batch_size, time_bins):
    total = 0
    for key in time_bins.keys():
        val = time_bins[key]
        total += len(val) // batch_size + 1
    return total


def build_dataloader(
    datalist,
    text_cleaner,
    batch_size,
    collate_config={},
    *,
    config,
    model_config,
):
    dataset = FilePathDataset(
        data_list=datalist,
        root_path=Path(config.dataset.path) / config.dataset.wav_path,
        text_cleaner=text_cleaner,
        model_config=model_config,
        pitch_path="",
        alignment_path="",
        duration_processor=None,
    )

    collate_fn = Collater(
        stage="alignment", hop_length=model_config.hop_length, **collate_config
    )
    time_bins, _ = dataset.time_bins()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        pin_memory=False,
        batch_sampler=DynamicBatchSampler(
            time_bins,
            shuffle=False,
            drop_last=False,
            force_bin=None,
            force_batch_size=batch_size,
            train=None,
        ),
    )
    total_steps = get_steps(batch_size, time_bins)
    return data_loader, total_steps


def align_text(config, model_config, method, batch_size):
    root = Path(config.dataset.path)

    out = root / config.dataset.alignment_path
    model = root / config.dataset.alignment_model_path
    device = config.training.device
    if device == "mps":
        device = "cpu"
        logger.info(
            f"Alignment does not support mps device. Falling back on cpu training."
        )

    # TrainContext requires an output directiary to save tensorboard, normalization, etc
    stage = "temp"
    train = TrainContext(stage, root, config, model_config, logger)
    train.batch_manager = BatchManager(
        train.config.dataset,
        train.out_dir,
        probe_batch_max=1,
        device=train.config.training.device,
        accelerator=train.accelerator,
        multispeaker=train.model_config.multispeaker,
        text_cleaner=train.text_cleaner,
        stage=stage,
        epoch=train.manifest.current_epoch,
        train=train,
    )
    train.init_normalization()
    train.to_align_mel = train.to_align_mel.to(device)

    aligner_dict = load_file(model, device=device)
    aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.tokens
    )
    aligner = aligner.to(device)
    aligner.load_state_dict(aligner_dict)
    aligner = aligner.eval()

    if method == "k2":
        val_datalist = get_data_path_list(
            Path(config.dataset.path) / config.dataset.val_data
        )
        vals, scores = calculate_alignment_batched(
            train, "Val Set", val_datalist, batch_size, config, model_config, aligner
        )
        with open(
            Path(config.dataset.path) / "scores_val.txt", "w", encoding="utf-8"
        ) as f:
            for name in scores.keys():
                f.write(str(scores[name]) + " " + name + "\n")

        train_datalist = get_data_path_list(
            Path(config.dataset.path) / config.dataset.train_data
        )
        trains, scores = calculate_alignment_batched(
            train,
            "Train Set",
            train_datalist,
            batch_size,
            config,
            model_config,
            aligner,
        )
        with open(
            Path(config.dataset.path) / "scores_train.txt", "w", encoding="utf-8"
        ) as f:
            for name in scores.keys():
                f.write(str(scores[name]) + " " + name + "\n")
    elif method == "torch":
        wavdir = root / config.dataset.wav_path
        vals, scores = calculate_alignments(
            train,
            "Val Set",
            root / config.dataset.val_data,
            wavdir,
            aligner,
            model_config,
            device,
        )
        with open(
            Path(config.dataset.path) / "scores_val.txt", "w", encoding="utf-8"
        ) as f:
            for name in scores.keys():
                f.write(str(scores[name]) + " " + name + "\n")
        trains, scores = calculate_alignments(
            train,
            "Train Set",
            root / config.dataset.train_data,
            wavdir,
            aligner,
            model_config,
            device,
        )
        with open(
            Path(config.dataset.path) / "scores_train.txt", "w", encoding="utf-8"
        ) as f:
            for name in scores.keys():
                f.write(str(scores[name]) + " " + name + "\n")
    else:
        raise NotImplementedError(method)
    result = vals | trains
    if out.exists():
        # Fix safetensors_rust.SafetensorError:
        # Error while serializing: I/O error:
        # The requested operation cannot be performed on a file with a user-mapped section open.
        # (os error 1224)
        out.unlink()
    save_file(result, out)
    # Remove temporary output directoary
    shutil.rmtree(train.out_dir)


def tqdm_wrapper(iterable, total=None, desc="", color="GREEN"):
    return tqdm.tqdm(
        iterable=iterable,
        desc=desc,
        unit="segments",
        initial=0,
        colour=color,
        dynamic_ncols=True,
        total=total,
    )


@torch.no_grad()
def calculate_alignment_batched(
    train: TrainContext, label, datalist, batch_size, config, model_config, aligner
):
    alignment_map = {}
    scores_map = {}

    dataloader, total_steps = build_dataloader(
        datalist,
        train.text_cleaner,
        batch_size,
        config=config,
        model_config=model_config,
    )
    iterator = tqdm_wrapper(
        dataloader,
        total=total_steps,
        desc="Aligning " + label,
        color="MAGENTA",
    )
    for inputs in iterator:
        batch = prepare_batch(
            inputs,
            train.config.training.device,
            ["audio_gt", "text", "text_length", "path"],
        )
        mels, mel_lengths = calculate_mel(
            batch.audio_gt,
            train.to_align_mel,
            train.normalization.mel_log_mean,
            train.normalization.mel_log_std,
        )
        mels = rearrange(mels, "b f t -> b t f")
        prediction, _ = aligner(mels, mel_lengths)
        prediction = rearrange(prediction, "t b k -> b t k")
        alignments, scores = k2_align(
            batch.text, mel_lengths, batch.text_length, prediction, train
        )
        for name, alignment, score in zip(batch.path, alignments, scores):
            alignment_map[name] = alignment
            scores_map[name] = score.exp().item()
    return alignment_map, scores_map


@torch.no_grad()
def calculate_alignments(
    train,
    label,
    path,
    wavdir,
    aligner,
    model_config,
    device,
):
    alignment_map = {}
    scores_map = {}

    with path.open("r", encoding="utf-8") as f:
        total_segments = sum(1 for _ in f)
    iterator = tqdm_wrapper(
        audio_list(path, wavdir, model_config),
        total=total_segments,
        desc="Aligning " + label,
        color="MAGENTA",
    )
    for name, text_raw, wave in iterator:
        alignment, scores = calculate_alignment_single(
            train, aligner, model_config, name, text_raw, wave, device
        )
        alignment_map[name] = alignment
        scores_map[name] = scores.exp().mean().item()
    return alignment_map, scores_map


def calculate_alignment_single(
    train: TrainContext, aligner, model_config, name, text, wave, device
):
    mels, mel_lengths = calculate_mel(
        torch.from_numpy(wave).float().to(device).unsqueeze(0),
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    text = train.text_cleaner(text)
    text = torch.tensor(text).to(device).unsqueeze(0)
    mels = rearrange(mels, "b f t -> b t f")
    prediction, _ = aligner(mels, mel_lengths)
    prediction = rearrange(prediction, "t b k -> b t k")

    text_lengths = torch.zeros([1], dtype=int, device=device)
    text_lengths[0] = text.shape[1]

    alignment, scores = torch_align(
        mels, text, mel_lengths, text_lengths, prediction, model_config, name
    )
    # alignment = teytaut_align(mels, text, mel_lengths, text_lengths, prediction)
    return alignment, scores


def torch_align(mels, text, mel_length, text_length, prediction, model_config, name):
    blank = model_config.text_encoder.tokens
    alignment, scores = torchaudio.functional.forced_align(
        log_probs=prediction,
        targets=text,
        input_lengths=mel_length,
        target_lengths=text_length,
        blank=blank,
    )
    alignment = alignment.squeeze()
    atensor = torch.zeros(
        [1, text.shape[1], alignment.shape[0]], device=mels.device, dtype=bool
    )
    text_index = 0
    last_text = alignment[0]
    was_blank = False
    for i in range(alignment.shape[0]):
        if alignment[i] == blank:
            was_blank = True
        else:
            if alignment[i] != last_text or was_blank:
                text_index += 1
                last_text = alignment[i]
                was_blank = False
        if text_index >= text.shape[-1]:
            print(
                "WARNING: alignment is longer than the sequence, likely an untrained model."
            )
            break
        if alignment[i] == blank or alignment[i] == text[0, text_index]:
            atensor[0, text_index, i] = 1
        else:
            print(
                "WARNING: the alignment doesn't match the sequence, likely an untrained model."
            )
    pred_dur = atensor.sum(dim=2).squeeze(0)
    result = torch.zeros(
        [1, pred_dur.shape[0]], dtype=torch.float, device=pred_dur.device
    )
    result[0] = pred_dur
    # pred_dur = torch.nn.functional.pad(pred_dur, (3, 3))
    # pred_dur = pred_dur.cpu().numpy()
    # end_indices = pred_dur.cumsum() + 3
    # begin_indices = end_indices - pred_dur
    # mean_list = (begin_indices + end_indices - 1) / 2

    # prediction = prediction.transpose(1, 2)
    # prediction = torch.nn.functional.pad(prediction, (3, 3), value=-1e9)
    # prediction = prediction.transpose(1, 2)

    # for i in range(3, pred_dur.size - 3):
    #     begin = begin_indices[i] - 3
    #     end = end_indices[i] + 3
    #     center = mean_list[i]
    #     y_data = prediction[0, begin:end, text[0, i-3]].exp().cpu().numpy()
    #     y_data[0] = 0.0
    #     y_data[-1] = 0.0
    #     # y_max = y_data.max()
    #     # y_data = y_data / (y_max + 1e-9)
    #     x_data = numpy.arange(begin, end) - center
    #     series = numpy.polynomial.polynomial.Polynomial.fit(
    #         x=x_data,
    #         y=y_data,
    #         deg=4,
    #         rcond=0.05,
    #     )
    #     # series = series.convert()
    #     for j in range(5):
    #         result[j+1, i-3] = series.coef[j]
    return result, scores
    # left = torch.zeros_like(pred_dur, dtype=torch.float)
    # right = torch.zeros_like(pred_dur, dtype=torch.float)
    # index = 0
    # for i in range(pred_dur.shape[0] - 1):
    #     index += pred_dur[i]
    #     left_token = text[0, i]
    #     right_token = text[0, i + 1]
    #     left_prob = math.exp(
    #         prediction[0, index - 1, left_token] + prediction[0, index, left_token]
    #     )
    #     split_prob = math.exp(
    #         prediction[0, index - 1, left_token] + prediction[0, index, right_token]
    #     )
    #     right_prob = math.exp(
    #         prediction[0, index - 1, right_token] + prediction[0, index, right_token]
    #     )
    #     denom = left_prob + split_prob + right_prob
    #     left[i] = left_prob / denom
    #     right[i] = right_prob / denom
    # return torch.stack([pred_dur, left, right]), scores


def k2_align(text, mel_length, text_length, prediction, train):
    batch_frame_labels, scores = train.align_loss.forced_align(
        log_probs=prediction,
        targets=text,
        input_lengths=mel_length,
        target_lengths=text_length,
    )
    # k2 forces blank to be 0, 
    # therefore the prefix, suffix pad tokens are collapsed into first, last tokens, respectfully.
    # The predicted labels w.r.t audio frames are used to guess the duration of pad tokens.
    batch_frame_labels_pred = prediction.argmax(dim=-1)
    durations = []
    for i, seq_frames in enumerate(batch_frame_labels):
        total_frames = len(batch_frame_labels_pred[i])
        token_indices = [idx for idx, label in enumerate(seq_frames) if label > 0]
        if not token_indices:
            print("WARNING: no tokens found, likely an untrained model.")
            durations.append(torch.tensor([total_frames]))
            continue

        first_idx = token_indices[0]
        last_idx = token_indices[-1]

        # The index of the first token is exactly the number of silence frames before it.
        prefix_dur = first_idx
        token_durs = []

        # Process from first token up to (but not including) the last token
        current_dur = 0

        # Slice stops before last_idx to handle the last token specially
        for label in seq_frames[first_idx:last_idx]:
            if label > 0:
                if current_dur > 0:
                    token_durs.append(current_dur)
                current_dur = 1  # Reset for new token
            else:
                current_dur += 1  # Add silence to current token

        # Append the duration of the token immediately preceding the last token
        if current_dur > 0 and len(token_indices) > 1:
            token_durs.append(current_dur)

        # Look at argmax starting from the last token's position
        last_token_activity = batch_frame_labels_pred[i, last_idx:]

        # Find first silence (0) in the argmax after the token starts
        silence_starts = (last_token_activity == 0).nonzero()

        if silence_starts.numel() > 0:
            last_token_dur = silence_starts[0].item()
            last_token_dur = max(1, last_token_dur)  # Ensure it has at least 1 frame
        else:
            # Token goes all the way to the end of the audio
            last_token_dur = len(last_token_activity)
        token_durs.append(last_token_dur)

        # Calculate where the speech actually ended
        speech_end_index = last_idx + last_token_dur

        # Remaining frames are the suffix
        suffix_dur = total_frames - speech_end_index

        # Clamp to 0 just in case of index misalignment (though rare)
        suffix_dur = max(0, suffix_dur)

        # Combine: [Prefix, tokens..., Suffix]
        full_durs = [prefix_dur] + token_durs + [suffix_dur]
        durations.append(torch.tensor(full_durs).unsqueeze(0))
    return durations, scores


def teytaut_align(mels, text, mel_length, text_length, prediction):
    # soft = soft_alignment(prediction, text)
    soft = soft_alignment_bad(prediction, text)
    soft = rearrange(soft, "b t k -> b k t")
    mask_ST = mask_from_lens(soft, text_length, mel_length)
    duration = maximum_path(soft, mask_ST)
    return duration


def soft_alignment(pred, phonemes):
    """
    Args:
        pred (b t k): Predictions of k (+ blank) tokens at time frame t
        phonemes (b p): Target sequence of phonemes
        mask (b p): Mask for target sequence
    Returns:
        (b t p): Phoneme predictions for each time frame t
    """
    # mask = rearrange(mask, "b p -> b 1 p")
    # Convert to <blank>, <phoneme>, <blank> ...
    # blank_id = pred.shape[2] - 1
    # blanks = torch.full_like(phonemes, blank_id)
    # ph_blank = rearrange([phonemes, blanks], "n b p -> b (p n)")
    # ph_blank = F.pad(ph_blank, (0, 1), value=blank_id)
    # ph_blank = rearrange(ph_blank, "b p -> b 1 p")
    ph_blank = rearrange(phonemes, "b p -> b 1 p")
    pred = pred.softmax(dim=2)
    pred = pred[:, :, :-1]
    pred = F.normalize(input=pred, p=1, dim=2)
    probability = torch.take_along_dim(input=pred, indices=ph_blank, dim=2)

    base_case = torch.zeros_like(ph_blank, dtype=pred.dtype).to(pred.device)
    base_case[:, :, 0] = 1
    result = [base_case]
    prev = base_case

    # Now everything should be (b t p)
    for i in range(1, probability.shape[1]):
        p0 = prev
        p1 = F.pad(prev[:, :, :-1], (1, 0), value=0)
        # p2 = F.pad(prev[:, :, :-2], (2, 0), value=0)
        # p2_mask = torch.not_equal(ph_blank, blank_id)
        prob = probability[:, i, :]
        prob = rearrange(prob, "b p -> b 1 p")
        # prev = (p0 + p1 + p2 * p2_mask) * prob
        prev = (p0 + p1) * prob
        prev = F.normalize(input=prev, p=1, dim=2)
        result.append(prev)
    result = torch.cat(result, dim=1)
    # unblank_indices = torch.arange(
    #     0, result.shape[2], 2, dtype=int, device=result.device
    # )
    # result = torch.index_select(input=result, dim=2, index=unblank_indices)
    # result = F.normalize(input=result, p=1, dim=2)
    result = (result + 1e-12).log()
    # result = result * ~mask
    return result


def soft_alignment_bad(pred, phonemes):
    """
    Args:
        pred (b t k): Predictions of k (+ blank) tokens at time frame t
        phonemes (b p): Target sequence of phonemes
        mask (b p): Mask for target sequence
    Returns:
        (b t p): Phoneme predictions for each time frame t
    """
    # mask = rearrange(mask, "b p -> b 1 p")
    # Convert to <blank>, <phoneme>, <blank> ...
    # blank_id = pred.shape[2] - 1
    # blanks = torch.full_like(phonemes, blank_id)
    # ph_blank = rearrange([phonemes, blanks], "n b p -> b (p n)")
    # ph_blank = F.pad(ph_blank, (0, 1), value=blank_id)
    # ph_blank = rearrange(ph_blank, "b p -> b 1 p")
    ph_blank = rearrange(phonemes, "b p -> b 1 p")
    # pred = pred.softmax(dim=2)
    pred = pred[:, :, :-1]
    pred = pred.exp()
    pred = torch.nn.functional.normalize(input=pred, p=1, dim=2)
    pred = pred.log()
    # pred = pred.log_softmax(dim=2)
    probability = torch.take_along_dim(input=pred, indices=ph_blank, dim=2)

    base_case = torch.full_like(ph_blank, fill_value=-math.inf, dtype=pred.dtype).to(
        pred.device
    )
    base_case[:, :, 0] = 0
    result = [base_case]
    prev = base_case

    # Now everything should be (b t p)
    for i in range(1, probability.shape[1]):
        p0 = prev
        p1 = torch.nn.functional.pad(prev[:, :, :-1], (1, 0), value=-math.inf)
        # p2 = F.pad(prev[:, :, :-2], (2, 0), value=0)
        # p2_mask = torch.not_equal(ph_blank, blank_id)
        prob = probability[:, i, :]
        prob = rearrange(prob, "b p -> b 1 p")
        # prev = (p0 + p1 + p2 * p2_mask) * prob
        prev = torch.logaddexp(p0, p1) + prob
        prev = prev.log_softmax(dim=2)
        # prev = torch.nn.functional.normalize(input=prev, p=1, dim=2)
        result.append(prev)
    result = torch.cat(result, dim=1)
    # unblank_indices = torch.arange(
    #     0, result.shape[2], 2, dtype=int, device=result.device
    # )
    # result = torch.index_select(input=result, dim=2, index=unblank_indices)
    # result = F.normalize(input=result, p=1, dim=2)
    # result = (result + 1e-12).log()
    # result = result * ~mask
    return result


def audio_list(path, wavdir, model_config):
    coarse_hop_length = model_config.hop_length * model_config.coarse_multiplier
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.split("|")
            name = fields[0]
            phonemes = fields[1]
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                sys.stderr.write(f"Skipping {name}: Wrong sample rate ({sr})")
            if wave.shape[-1] == 2:
                wave = wave[:, 0].squeeze()
            time_bin = get_time_bin(
                wave.shape[0], model_config.hop_length * model_config.coarse_multiplier
            )
            if time_bin == -1:
                sys.stderr.write(f"Skipping {name}: Too short\n")
                continue
            frame_count = get_frame_count(time_bin)
            pad_start = (frame_count * coarse_hop_length - wave.shape[0]) // 2
            pad_end = frame_count * coarse_hop_length - wave.shape[0] - pad_start
            wave = numpy.concatenate(
                [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
            )
            yield name, phonemes, wave
