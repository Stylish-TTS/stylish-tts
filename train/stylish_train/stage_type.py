import random
from typing import Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange
from loss_log import LossLog, build_loss_log
from losses import compute_duration_ce_loss
from utils import print_gpu_vram, log_norm, duration_to_alignment
from typing import List


stages = {}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class StageType:
    def __init__(
        self,
        next_stage: Optional[str],
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        discriminators: List[str],
        inputs: List[str],
    ):
        self.next_stage: Optional[str] = next_stage
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.discriminators = discriminators
        self.inputs: List[str] = inputs


def make_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    result = []
    for i in range(tensor.shape[0]):
        result.append(tensor[i])
    return result


##### Alignment #####


def train_alignment(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss())
    return log.detach(), None


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=batch.mel_length[i].unsqueeze(0),
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


stages["alignment"] = StageType(
    next_stage=None,
    train_fn=train_alignment,
    validate_fn=validate_alignment,
    train_models=["text_aligner"],
    eval_models=[],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "align_mel",
        "mel_length",
    ],
)

##### Acoustic #####


def train_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        print_gpu_vram("init")
        with torch.no_grad():
            energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, batch.pitch, energy
        )
        pred_pitch, pred_energy = model.pitch_energy_predictor(
            batch.text, batch.text_length, batch.alignment
        )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        print_gpu_vram("stft_loss")
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio, ["mrd"]
            ).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        train.magphase_loss(pred, batch.audio_gt, log)
        print_gpu_vram("magphase_loss")

        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        train.accelerator.backward(log.backwards_loss())
        print_gpu_vram("backward")

    return log.detach(), pred.audio.detach()


@torch.no_grad()
def validate_acoustic(batch, train):
    energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        batch.text, batch.text_length, batch.alignment
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, batch.pitch, energy
    )
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss(
        "energy",
        torch.nn.functional.smooth_l1_loss(energy, pred_energy),
    )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["acoustic"] = StageType(
    next_stage="textual",
    train_fn=train_acoustic,
    validate_fn=validate_acoustic,
    train_models=["speech_predictor", "pitch_energy_predictor"],
    eval_models=[],
    discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "mel",
        "mel_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Textual #####


def train_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        pred_pitch, pred_energy = model.pitch_energy_predictor(
            batch.text, batch.text_length, batch.alignment
        )
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
        )
        with torch.no_grad():
            energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio, ["mrd"]
            ).mean(),
        )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        train.accelerator.backward(log.backwards_loss())

    return log.detach(), pred.audio.detach()


@torch.no_grad()
def validate_textual(batch, train):
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        batch.text, batch.text_length, batch.alignment
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
    )
    energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["textual"] = StageType(
    next_stage="duration",
    train_fn=train_textual,
    validate_fn=validate_textual,
    train_models=["pitch_energy_predictor"],
    eval_models=["speech_predictor"],
    discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "mel",
        "mel_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Duration #####


def train_duration(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    duration = model.duration_predictor(batch.text, batch.text_length)
    train.stage.optimizer.zero_grad()
    log = build_loss_log(train)
    loss_ce, loss_dur = compute_duration_ce_loss(
        duration,
        batch.alignment.sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    train.accelerator.backward(log.backwards_loss())

    return log.detach(), None


@torch.no_grad()
def validate_duration(batch, train):
    duration = train.model.duration_predictor(batch.text, batch.text_length)
    results = []
    for i in range(duration.shape[0]):
        dur = torch.sigmoid(duration[i]).sum(axis=-1)
        dur = torch.round(dur).clamp(min=1).long()
        dur = dur[: batch.text_length[i]]
        alignment = duration_to_alignment(dur)
        alignment = rearrange(alignment, "t a -> 1 t a")
        pred_pitch, pred_energy = train.model.pitch_energy_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
        )
        kernel = torch.FloatTensor(
            [[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]
        ).to(batch.text.device)
        pred_pitch = torch.nn.functional.conv1d(pred_pitch, kernel, padding=3)
        pred_energy = torch.nn.functional.conv1d(pred_energy, kernel, padding=3)
        pred = train.model.speech_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
            pred_pitch,
            pred_energy,
        )
        audio = rearrange(pred.audio, "1 1 l -> l")
        results.append(audio)
    log = build_loss_log(train)
    loss_ce, loss_dur = compute_duration_ce_loss(
        duration,
        batch.alignment.sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)

    return log.detach(), alignment[0], results, batch.audio_gt


stages["duration"] = StageType(
    next_stage=None,
    train_fn=train_duration,
    validate_fn=validate_duration,
    train_models=[
        "duration_predictor",
    ],
    eval_models=["pitch_energy_predictor", "speech_predictor"],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "alignment",
        "audio_gt",
    ],
)
