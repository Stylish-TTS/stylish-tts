import math
from typing import Optional, Tuple
import torch
from einops import rearrange
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log
from losses import magphase_loss, compute_duration_ce_loss, freev_loss
from utils import length_to_mask


def train_alignment(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)

    # blank = train.text_cleaner("ǁ")[0]
    # mask = length_to_mask(batch.mel_length // 2).to(train.config.training.device)
    # ppgs, s2s_pred, _ = model.text_aligner(
    #     batch.mel, src_key_padding_mask=mask, text_input=batch.text
    # )

    # ctc = (b t k), reconstruction = (b f t)
    # ctc, reconstruction = model.text_aligner(batch.mel)
    mel = rearrange(batch.mel, "b f t -> b t f")
    ctc = model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length // 2, batch.text_length, step_type="train"
    )

    # ctc = rearrange(ctc, "b t k -> t b k")
    # softlog = ctc.log_softmax(dim=2)
    # loss_ctc = torch.nn.functional.ctc_loss(
    #     softlog,
    #     batch.text,
    #     batch.mel_length,
    #     batch.text_length,
    #     blank=train.model_config.text_encoder.n_token,
    # )
    # loss_ctc = train.align_loss(
    #     attn_logprob, in_lens=batch.text_length, out_lens=batch.mel_length
    # )
    log.add_loss(
        "align_loss",
        # loss_ctc + 0.1 * torch.nn.functional.l1_loss(reconstruction, batch.mel),
        loss_ctc,
    )
    # log.add_loss("align_ctc", loss_ctc)
    # log.add_loss("align_rec", 0.1 * torch.nn.functional.l1_loss(reconstruction, batch.mel))

    #     soft = ppgs.log_softmax(dim=2).transpose(0, 1)
    #     loss_ctc = torch.nn.functional.ctc_loss(
    #         soft,
    #         batch.text,
    #         batch.mel_length // 2,
    #         batch.text_length,
    #         blank=blank,
    #     )
    #     log.add_loss("ctc", loss_ctc)
    #
    #     loss_s2s = 0
    #     for pred_align, text, length in zip(s2s_pred, batch.text, batch.text_length):
    #         loss_s2s += torch.nn.functional.cross_entropy(
    #             pred_align[:length], text[:length], ignore_index=-1
    #         )
    #     loss_s2s /= batch.text.size(0)
    #     log.add_loss("s2s", loss_s2s)

    train.accelerator.backward(log.backwards_loss() * math.sqrt(batch.text.shape[0]))
    return log.detach(), None


def train_pre_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch, use_random_mono=True)
        with torch.no_grad():
            counter = state.acoustic_prediction_single(
                batch, use_random_mono=True, switch_style=True
            )
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        repulse_loss = train.stft_loss(
            pred.audio.squeeze(1), counter.audio.squeeze(1), log=None
        )
        log.add_loss("repulse", -repulse_loss)
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
    freev_loss(log, pred, batch.audio_gt, train)
    train.accelerator.backward(log.backwards_loss() * math.sqrt(batch.text.shape[0]))
    return log.detach(), pred.audio.detach()


def train_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch)
        with torch.no_grad():
            counter = state.acoustic_prediction_single(batch, switch_style=True)
        # train.stage.optimizer.zero_grad()
        # d_loss = train.discriminator_loss(
        #     batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        # ).mean()
        # train.accelerator.backward(d_loss)
        # train.stage.optimizer.step("msd")
        # train.stage.optimizer.step("mpd")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        repulse_loss = train.stft_loss(
            pred.audio.squeeze(1), counter.audio.squeeze(1), log=None
        )
        log.add_loss("repulse", -repulse_loss)
        # d_index = 0
        # if not probing:
        # d_index = train.manifest.current_total_step % 4
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(),
                pred.audio,
            ).mean(),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )

        loss_s2s = 0
        #         for pred_align, text, length in zip(
        #             state.s2s_pred, batch.text, batch.text_length
        #         ):
        #             loss_s2s += torch.nn.functional.cross_entropy(
        #                 pred_align[:length], text[:length]
        #             )
        #         loss_s2s /= batch.text.size(0)
        #         log.add_loss("s2s", loss_s2s)
        #
        #         log.add_loss(
        #             "mono", torch.nn.functional.l1_loss(*(state.duration_results)) * 10
        #         )

        freev_loss(log, pred, batch.audio_gt, train)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )
        # log.add_loss("discriminator", d_loss)

    return log.detach(), pred.audio.detach()


def train_pre_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        state.textual_bootstrap_prediction(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        loss_ce, loss_dur = compute_duration_ce_loss(
            state.duration_prediction,
            state.duration_results[1].sum(dim=-1),
            batch.text_length,
        )
        log.add_loss("duration_ce", loss_ce)
        log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), None


def train_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        loss_ce, loss_dur = compute_duration_ce_loss(
            state.duration_prediction,
            state.duration_results[1].sum(dim=-1),
            batch.text_length,
        )
        log.add_loss("duration_ce", loss_ce)
        log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), pred.audio.detach()


def train_joint(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        # train.stage.optimizer.zero_grad()
        # d_loss = train.discriminator_loss(
        #     batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        # ).mean()
        # train.accelerator.backward(d_loss)
        # train.stage.optimizer.step("msd")
        # train.stage.optimizer.step("mpd")
        # train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio
            ).mean(),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        loss_ce, loss_dur = compute_duration_ce_loss(
            state.duration_prediction,
            state.duration_results[1].sum(dim=-1),
            batch.text_length,
        )
        log.add_loss("duration_ce", loss_ce)
        log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )
        # log.add_loss("discriminator", d_loss)

    return log.detach(), pred.audio.detach()


def train_sbert(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Training function for the sbert stage."""
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        # 1. Get textual and acoustic embeddings
        textual_style_embedding = state.textual_style_embedding(
            batch.sentence_embedding
        )
        textual_prosody_embedding = state.textual_prosody_embedding(
            batch.sentence_embedding
        )
        acoustic_style_embedding = state.acoustic_style_embedding(batch.mel)
        acoustic_prosody_embedding = state.acoustic_prosody_embedding(batch.mel)

        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)

        # 2. Calculate Loss
        style_loss = torch.nn.functional.l1_loss(
            textual_style_embedding, acoustic_style_embedding
        )
        prosody_loss = torch.nn.functional.l1_loss(
            textual_prosody_embedding, acoustic_prosody_embedding
        )

        log.add_loss("sbert_style_loss", style_loss)
        log.add_loss("sbert_prosody_loss", prosody_loss)

        train.accelerator.backward(log.total())

    return log.detach(), None
