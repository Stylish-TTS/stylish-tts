import click
import logging
import math
from os import path as osp
import pathlib
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
from stylish_tts.train.dataloader import get_frame_count, get_time_bin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# TODO: Merge this to_mel implementation with the one in stage_type
to_mel = None
norm_mean = -4.0
norm_std = 4.0


def align_text(config, model_config):
    root = pathlib.Path(config.dataset.path)

    out = root / config.dataset.alignment_path
    model = root / config.dataset.alignment_model_path
    device = config.training.device
    if device == "mps":
        device = "cpu"
        logger.info(
            f"Alignment does not support mps device. Falling back on cpu training."
        )

    global to_mel, norm_mean, norm_std

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80,  # align seems to perform worse on higher n_mels
        n_fft=model_config.n_fft,
        win_length=model_config.win_length,
        hop_length=model_config.hop_length,
        sample_rate=model_config.sample_rate,
    )

    # Try to load dataset normalization stats if available
    try:
        import json

        stats_path = pathlib.Path(config.dataset.path) / "normalization.json"
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            norm_mean = float(data.get("mel_log_mean", -4.0))
            norm_std = float(data.get("mel_log_std", 4.0))
            logger.info(
                f"Using dataset normalization stats for alignment: mean={norm_mean:.4f}, std={norm_std:.4f}"
            )
        else:
            logger.warning(
                "Dataset normalization.json not found; using default normalization (-4, 4) for alignment."
            )
    except Exception as e:
        logger.warning(f"Could not load dataset normalization.json: {e}")

    aligner_dict = load_file(model, device=device)
    aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.tokens
    )
    aligner = aligner.to(device)
    aligner.load_state_dict(aligner_dict)
    aligner = aligner.eval()

    text_cleaner = TextCleaner(model_config.symbol)

    wavdir = root / config.dataset.wav_path
    vals, scores = calculate_alignments(
        "Val Set",
        root / config.dataset.val_data,
        wavdir,
        aligner,
        model_config,
        text_cleaner,
        device,
    )
    with open(pathlib.Path(config.dataset.path) / "scores_val.txt", "w", encoding="utf-8") as f:
        for name in scores.keys():
            f.write(str(scores[name]) + " " + name + "\n")
    trains, scores = calculate_alignments(
        "Train Set",
        root / config.dataset.train_data,
        wavdir,
        aligner,
        model_config,
        text_cleaner,
        device,
    )
    with open(pathlib.Path(config.dataset.path) / "scores_train.txt", "w", encoding="utf-8") as f:
        for name in scores.keys():
            f.write(str(scores[name]) + " " + name + "\n")
    result = vals | trains
    save_file(result, out)


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - norm_mean) / norm_std
    mel_tensor = mel_tensor[:, :, :-1]
    return mel_tensor


@torch.no_grad()
def calculate_alignments(
    label, path, wavdir, aligner, model_config, text_cleaner, device
):
    with path.open("r", encoding="utf-8") as f:
        total_segments = sum(1 for _ in f)
    alignment_map = {}
    scores_map = {}
    iterator = tqdm.tqdm(
        iterable=audio_list(path, wavdir, model_config),
        desc="Aligning " + label,
        unit="segments",
        initial=0,
        colour="MAGENTA",
        dynamic_ncols=True,
        total=total_segments,
    )
    for name, text_raw, wave in iterator:
        mels = preprocess(wave).to(device)
        text = text_cleaner("$" + text_raw + "$")
        text = torch.tensor(text).to(device).unsqueeze(0)
        mels = rearrange(mels, "b f t -> b t f")
        mel_lengths = torch.zeros([1], dtype=int, device=device)
        mel_lengths[0] = mels.shape[1]
        prediction, _ = aligner(mels, mel_lengths)
        prediction = rearrange(prediction, "t b k -> b t k")

        text_lengths = torch.zeros([1], dtype=int, device=device)
        text_lengths[0] = text.shape[1]

        alignment, scores = torch_align(
            mels, text, mel_lengths, text_lengths, prediction, model_config, name
        )
        # alignment = teytaut_align(mels, text, mel_lengths, text_lengths, prediction)
        alignment_map[name] = alignment
        scores_map[name] = scores.exp().mean().item()
    return alignment_map, scores_map


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
