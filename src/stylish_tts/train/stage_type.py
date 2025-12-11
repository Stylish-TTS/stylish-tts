import math
import random
from typing import Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchaudio
from einops import rearrange
from stylish_tts.train.loss_log import LossLog, build_loss_log
from stylish_tts.train.utils import (
    print_gpu_vram,
    log_norm,
    length_to_mask,
    leaky_clamp,
    calculate_mel,
    normalize_log2,
    denormalize_log2,
)
from typing import List
from stylish_tts.train.losses import multi_phase_loss
import numpy as np

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


class AcousticStep:
    def __init__(
        self,
        batch,
        train,
        loss_log,
        alignment=None,
        *,
        use_predicted_pe,
        predict_audio,
    ):
        self.batch = batch
        self.train = train
        self.log = loss_log
        with torch.no_grad():
            self.mel, _ = calculate_mel(
                batch.audio_gt,
                train.to_mel,
                train.normalization.mel_log_mean,
                train.normalization.mel_log_std,
            )
            self.style_mel, _ = calculate_mel(
                batch.audio_gt,
                train.to_style_mel,
                train.normalization.mel_log_mean,
                train.normalization.mel_log_std,
            )
            self.energy = log_norm(
                self.mel.unsqueeze(1),
                train.normalization.mel_log_mean,
                train.normalization.mel_log_std,
            ).squeeze(1)
            self.voiced = (batch.pitch > 10).float()
            self.pitch = batch.pitch
            # self.target_pitch = batch.pitch
            # self.target_energy = self.energy
            self.energy = torch.log(self.energy + 1e-9)

            # self.energy = normalize_log2(
            #     self.energy,
            #     train.normalization.energy_log2_mean,
            #     train.normalization.energy_log2_std,
            # )
            # self.pitch = normalize_log2(
            #     self.pitch, train.f0_log2_mean, train.f0_log2_std
            # )

        if alignment is None:
            alignment = train.duration_processor.duration_to_alignment(
                batch.alignment[:, 0, :].long()
            )
            alignment_fine = train.duration_processor.duration_to_alignment(
                batch.alignment[:, 0, :].long(),
                multiplier=train.model_config.coarse_multiplier,
            )
        if use_predicted_pe:
            self.pe_style = train.model.pe_style_encoder(
                self.style_mel, self.pitch, self.energy
            )
            self.pred_pitch, self.pred_energy = train.model.pitch_energy_predictor(
                batch.text,
                batch.text_length,
                alignment,
                self.pe_style,
            )

            self.pitchcat = torch.stack([self.pitch * self.voiced, self.energy], dim=1)
            self.pred_pitchcat = torch.stack(
                [self.pred_pitch * self.voiced, self.pred_energy], dim=1
            )

        if predict_audio:
            self.speech_style = train.model.speech_style_encoder(
                self.style_mel.unsqueeze(1)
            )
            # voiced = self.voiced
            pitch = self.pitch
            energy = self.energy
            # base_pitch = batch.pitch
            if use_predicted_pe:
                pitch = self.pred_pitch
                energy = self.pred_energy
                # voiced = self.pred_voiced.round()
                # base_pitch = pitch
            # base_pitch = denormalize_log2(
            #     pitch, train.f0_log2_mean, train.f0_log2_std
            # )
            base_pitch = pitch
            voiced = (pitch > 20).float()
            # pitch = torch.log(torch.abs(pitch) + 1)
            # pitch = pitch * voiced
            # base_pitch = base_pitch * voiced
            self.pred = train.model.speech_predictor(
                batch.text,
                batch.text_length,
                alignment_fine,
                pitch,
                energy,
                voiced,
                self.speech_style,
                base_pitch,
            )
            (
                self.target_spec,
                self.pred_spec,
                self.target_phase,
                self.pred_phase,
                self.target_fft,
                self.pred_fft,
            ) = train.multi_spectrogram(
                target=batch.audio_gt, pred=self.pred.audio.squeeze(1)
            )
        else:
            self.pred = None
            self.target_spec = None
            self.pred_spec = None
            self.target_phase = None
            self.pred_phase = None
            self.target_fft = None
            self.pred_fft = None

    def mel_loss(self):
        self.train.stft_loss(
            target_list=self.target_spec, pred_list=self.pred_spec, log=self.log
        )

    def multi_phase_loss(self):
        self.log.add_loss(
            "multi_phase",
            multi_phase_loss(
                self.pred_phase, self.target_phase, self.train.model_config.n_fft
            ),
        )

    def pitch_generator_loss(self):
        self.log.add_loss(
            "generator",
            self.train.generator_loss(
                target_list=[self.pitchcat],
                pred_list=[self.pred_pitchcat],
                used=["pitch_disc"],
                index=0,
            ).mean(),
        )

    def generator_loss(self, disc_index):
        self.log.add_loss(
            "generator",
            self.train.generator_loss(
                target_list=self.target_fft,
                pred_list=self.pred_fft,
                used=["mrd"],
                index=disc_index,
            ).mean(),
        )

    def slm_loss(self):
        self.log.add_loss(
            "slm",
            self.train.wavlm_loss(self.batch.audio_gt.detach(), self.pred.audio),
        )

    def magphase_loss(self):
        self.train.magphase_loss(self.pred, self.batch.audio_gt, self.log)

    def pitch_loss(self):
        # target = torch.complex(self.pitch * self.voiced, self.energy*4)
        # prediction = torch.complex(self.pred_pitch * self.voiced, self.pred_energy*4)
        # self.log.add_loss(
        #     "pitch",
        #     torch.nn.functional.l1_loss(target, prediction),
        # )
        target = self.pitch
        prediction = self.pred_pitch
        self.log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(target, prediction)
            + torch.nn.functional.smooth_l1_loss(
                torch.diff(target), torch.diff(prediction)
            ),
        )
        target = self.energy
        prediction = self.pred_energy
        self.log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(target, prediction)
            + torch.nn.functional.smooth_l1_loss(
                torch.diff(target), torch.diff(prediction)
            ),
        )

    def voiced_loss(self):
        pass
        # self.log.add_loss(
        #     "voiced",
        #     torch.nn.functional.binary_cross_entropy(self.pred_voiced, self.voiced),
        # )


##### Alignment #####


def train_alignment(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(
        batch.audio_gt,
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss())
    return log.detach(), None, None


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(
        batch.audio_gt,
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=mel_length[i].unsqueeze(0),
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
        "audio_gt",
    ],
)

##### Acoustic #####


def train_acoustic(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the acoustic stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=False,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.generator_loss(disc_index)
    step.slm_loss()
    step.magphase_loss()

    train.accelerator.backward(log.backwards_loss())
    return log.detach(), detach_all(step.target_fft), detach_all(step.pred_fft)


@torch.no_grad()
def validate_acoustic(batch, train):
    """Validate a single batch for the acoustic stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=False,
        predict_audio=True,
    )

    step.mel_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["acoustic"] = StageType(
    next_stage="textual",
    train_fn=train_acoustic,
    validate_fn=validate_acoustic,
    train_models=[
        "speech_predictor",
        "speech_style_encoder",
    ],
    eval_models=[],
    discriminators=["mrd0", "mrd1", "mrd2"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Textual #####


def train_textual(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the textual stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    # step.pitch_generator_loss()
    step.pitch_loss()
    step.voiced_loss()

    train.accelerator.backward(log.backwards_loss())
    return (
        log.detach(),
        None,
        None,
    )  # detach_all([step.pitchcat]), detach_all([step.pred_pitchcat])


@torch.no_grad()
def validate_textual(batch, train):
    """Validate a single batch for the textual stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.pitch_loss()
    step.voiced_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["textual"] = StageType(
    next_stage="duration",
    train_fn=train_textual,
    validate_fn=validate_textual,
    train_models=[
        "pitch_energy_predictor",
        "pe_style_encoder",
    ],
    eval_models=["speech_predictor", "speech_style_encoder"],
    # discriminators=["pitch_disc"],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
        "path",
    ],
)

##### Duration #####


def train_duration(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    style_mel, _ = calculate_mel(
        batch.audio_gt,
        train.to_style_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )

    target_dur = batch.alignment[:, 0, :].long()
    targets = train.duration_processor.dur_to_class(target_dur)
    duration_style = model.duration_style_encoder(style_mel.unsqueeze(1))
    duration_raw = model.duration_predictor(
        batch.text, batch.text_length, duration_style
    )
    total_dur = batch.pitch.shape[-1]
    duration = train.duration_processor.prediction_to_duration(
        duration_raw, batch.text_length
    )

    train.stage.optimizer.zero_grad()
    duration_loss = 0
    for i in range(duration.shape[0]):
        duration_loss += F.smooth_l1_loss(
            duration[i, : batch.text_length[i]], target_dur[i, : batch.text_length[i]]
        )
    duration_loss /= duration.shape[0]

    duration_sums = duration.sum(dim=-1)
    duration_sum_target = torch.full_like(duration_sums, total_dur)

    loss_ce, loss_cdw = train.duration_loss(duration_raw, targets, batch.text_length)

    log = build_loss_log(train)
    target_disc = target_dur.float().unsqueeze(1)
    pred_disc = duration.unsqueeze(1)
    log.add_loss(
        "generator",
        train.generator_loss(
            target_list=[target_disc],
            pred_list=[pred_disc],
            used=["dur_disc"],
            index=0,
        ).mean(),
    )

    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", duration_loss)
    train.accelerator.backward(log.backwards_loss())

    return log.detach(), detach_all([target_disc]), detach_all([pred_disc])


@torch.no_grad()
def validate_duration(batch, train):
    energy_mel, _ = calculate_mel(
        batch.audio_gt,
        train.to_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    energy = log_norm(
        energy_mel.unsqueeze(1),
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    ).squeeze(1)
    energy = torch.log(energy + 1e-9)
    style_mel, _ = calculate_mel(
        batch.audio_gt,
        train.to_style_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    target_dur = batch.alignment[:, 0, :].long()
    targets = train.duration_processor.dur_to_class(target_dur)
    duration_style = train.model.duration_style_encoder(style_mel.unsqueeze(1))
    duration_raw = train.model.duration_predictor(
        batch.text, batch.text_length, duration_style
    )
    total_dur = target_dur.sum(-1).max()
    duration = train.duration_processor.prediction_to_duration(
        duration_raw, batch.text_length
    )

    pe_mel_style = train.model.pe_style_encoder(style_mel, batch.pitch, energy)
    speech_style = train.model.speech_style_encoder(style_mel.unsqueeze(1))

    results = []
    duration_loss = 0
    for i in range(duration.shape[0]):
        duration_loss += F.smooth_l1_loss(
            duration[i, : batch.text_length[i]], target_dur[i, : batch.text_length[i]]
        )

        alignment = train.duration_processor.duration_to_alignment(duration[i : i + 1])
        alignment_fine = train.duration_processor.duration_to_alignment(
            duration[i : i + 1], multiplier=train.model_config.coarse_multiplier
        )
        pred_pitch, pred_energy = train.model.pitch_energy_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment[:, : batch.text_length[i], :],
            pe_mel_style[i : i + 1],
        )
        # pred_voiced = pred_voiced.round()
        # base_pitch = denormalize_log2(pred_pitch, train.f0_log2_mean, train.f0_log2_std)
        pred_voiced = (pred_pitch > 20).float()
        pred = train.model.speech_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment_fine[:, : batch.text_length[i], :],
            pred_pitch,
            pred_energy,
            pred_voiced,
            speech_style[i : i + 1],
            pred_pitch,
        )
        audio = rearrange(pred.audio, "1 1 l -> l")
        results.append(audio)
    duration_loss /= duration.shape[0]
    log = build_loss_log(train)
    loss_ce, loss_cdw = train.duration_loss(
        duration_raw,
        targets,
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", duration_loss)  # loss_cdw)

    return log.detach(), alignment[0], results, batch.audio_gt


stages["duration"] = StageType(
    next_stage=None,
    train_fn=train_duration,
    validate_fn=validate_duration,
    train_models=[
        "duration_predictor",
        "duration_style_encoder",
    ],
    eval_models=[
        "pitch_energy_predictor",
        "pe_style_encoder",
        "speech_predictor",
        "speech_style_encoder",
    ],
    discriminators=["dur_disc"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

#########################


def detach_all(spec_list):
    result = []
    for item in spec_list:
        result.append(item.detach())
    return result
