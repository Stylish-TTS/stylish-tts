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
from safetensors.torch import load_file
import textgrid
import soundfile
import sys
import numpy
import librosa
import logging
from stylish_tts.train.dataprep.align_text import torch_align, k2_align

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_audio(model_config, audio_path):
    coarse_hop_length = model_config.hop_length * model_config.coarse_multiplier
    wave, _ = librosa.load(audio_path, sr=model_config.sample_rate)
    if wave.shape[-1] == 2:
        wave = wave[:, 0].squeeze()
    time_bin = get_time_bin(
        wave.shape[0], model_config.hop_length * model_config.coarse_multiplier
    )
    if time_bin == -1:
        sys.stderr.write(f"Skipping {audio_path}: Too short\n")
        return
    frame_count = get_frame_count(time_bin)
    pad_start = (frame_count * coarse_hop_length - wave.shape[0]) // 2
    pad_end = frame_count * coarse_hop_length - wave.shape[0] - pad_start
    wave = numpy.concatenate(
        [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
    )
    return wave


def align_textgrid(audio_path, text, config, model_config, method):
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
    aligner = tdnn_blstm_ctc_model_base(80, model_config.text_encoder.tokens)
    aligner = aligner.to(device)
    aligner.load_state_dict(aligner_dict)
    aligner = aligner.eval()
    process(train, aligner, audio_path, text, config, model_config, method, device)
    # Remove temporary output directoary
    # shutil.rmtree(train.out_dir)


def process(train, aligner, audio_path, text_raw, config, model_config, method, device):
    wave = load_audio(model_config, audio_path)
    audio_gt = torch.from_numpy(wave).float().to(device).unsqueeze(0)
    mels, mel_lengths = calculate_mel(
        audio_gt,
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    mels = rearrange(mels, "b f t -> b t f")
    # mels, _ = train.kanade_codec.get_ssl_embeddings(audio_gt)
    # mel_lengths = torch.tensor([mels.shape[1]] * mels.shape[0], device=mels.device)
    text = train.text_cleaner(text_raw)
    text = torch.tensor(text).to(device).unsqueeze(0)
    text_lengths = torch.zeros([1], dtype=int, device=device)
    text_lengths[0] = text.shape[1]

    # mels = rearrange(mels, "b f t -> b t f")
    prediction, _ = aligner(mels, mel_lengths)
    prediction: torch.Tensor = rearrange(prediction, "t b k -> b t k")
    topk = prediction[0].exp().topk(5, -1)
    for values, indices in zip(topk.values, topk.indices):
        print(
            *[
                train.text_cleaner.index_word_dictionary[i.item()] + f":{v}"
                for i, v in zip(indices, values)
            ]
        )
    print(
        "".join(
            [
                train.text_cleaner.index_word_dictionary[i.item()]
                for i in prediction[0].argmax(-1).unique_consecutive()
            ]
        )
    )

    coarse_hop_length: int = model_config.hop_length * model_config.coarse_multiplier
    hop_duration = 1 / (model_config.sample_rate / coarse_hop_length)
    if method == "k2":
        alignment, _ = k2_align(text, mel_lengths, text_lengths, prediction, train)
        alignment = alignment[0][0]
    elif method == "torch":
        alignment, _ = torch_align(
            mels, text, mel_lengths, text_lengths, prediction, model_config, audio_path
        )
        alignment = alignment[0]
    else:
        raise NotImplementedError(method)

    tg = textgrid.TextGrid()
    phone_tier = textgrid.IntervalTier(name="Phonemes")
    start_time = 0
    for token, duration in zip("$" + text_raw + "$", alignment):
        duration = duration.item() * hop_duration
        end_time = start_time + duration
        phone_tier.add(start_time, end_time, token)
        start_time += duration

    tg.extend([phone_tier])
    padded_audo_name = f"padded_{audio_path.split('/')[-1]}"
    textgrid_name = f"padded_{method}_{audio_path.split('/')[-1]}.textgrid"
    tg.write(textgrid_name)
    soundfile.write(padded_audo_name, wave, model_config.sample_rate)
    print(
        f'Open "{padded_audo_name}" and "{textgrid_name}" on Praat (https://www.fon.hum.uva.nl/praat/) to view the alignment.'
    )
