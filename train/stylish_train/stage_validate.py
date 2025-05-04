import random
import torch
import torchaudio
from torch.nn import functional as F
from einops import rearrange

from batch_context import BatchContext
from loss_log import build_loss_log
from losses import compute_duration_ce_loss
from utils import length_to_mask


@torch.no_grad()
def validate_text_encoder(batch, train):
    log = build_loss_log(train)

    submask = torch.ones(
        [batch.mel.shape[0], batch.mel.shape[2]], dtype=bool, device=batch.mel.device
    )
    for i in range(batch.mel.shape[2] // 100 + 1):
        substart = random.randrange(batch.mel.shape[2])
        subend = min(substart + 32, batch.mel.shape[2]) + 1
        # subend = random.randrange(substart + 4, min(substart + 16, batch.mel.shape[2])) + 1
        submask[:, substart:subend] = False
    submask = submask.unsqueeze(1)
    masked_mel = batch.mel * submask
    corrupted = train.model.text_mel_generator(masked_mel)
    corrupted = masked_mel + corrupted * ~submask
    text_spread = batch.text.unsqueeze(1).float() @ batch.alignment
    text_spread = F.interpolate(text_spread, scale_factor=2, mode="nearest-exact")
    text_spread = text_spread.squeeze(1).int()
    embedding = train.model.text_encoder(text_spread)
    embedding = rearrange(embedding, "b t c -> b c t")
    prediction = train.model.text_mel_classifier(corrupted.detach(), embedding)

    log.add_loss("text_gen", F.l1_loss(corrupted * ~submask, batch.mel * ~submask))
    disc_logits = prediction.squeeze(2)
    disc_labels = (~submask).float()
    disc_loss = F.binary_cross_entropy_with_logits(disc_logits, disc_labels)
    log.add_loss("text_disc", disc_loss)

    return log.detach(), None, None, None


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    # ctc, reconstruction = train.model.text_aligner(batch.mel)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    # ctc, reconstruction = train.model.text_aligner(mel, batch.mel_length)
    ctc, _ = train.model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()

    # softlog = ctc.log_softmax(dim=2)
    # softlog = ctc.transpose(0, 1)
    # loss_ctc = torch.nn.functional.ctc_loss(
    #     softlog,
    #     batch.text,
    #     batch.mel_length,
    #     batch.text_length,
    #     blank=train.model_config.text_encoder.n_token,
    # )
    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length // 2, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.n_token
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=batch.mel_length[i].unsqueeze(0) // 2,
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    # log.add_loss(
    #     "align_rec", torch.nn.functional.l1_loss(reconstruction, batch.align_mel)
    # )
    return log, None, None, None


#     blank = train.text_cleaner("ǁ")[0]
#     mask = length_to_mask(batch.mel_length // 2).to(train.config.training.device)
#     ppgs, s2s_pred, s2s_attn = train.model.text_aligner(
#         batch.mel, src_key_padding_mask=mask, text_input=batch.text
#     )
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
#     return log, s2s_attn[0], None, None


@torch.no_grad()
def validate_vocoder(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred, gt = state.audio_reconstruction(batch)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), gt, log)
    return log, None, pred.audio, gt


@torch.no_grad()
def validate_acoustic(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    return log, state.get_attention(), pred.audio, batch.audio_gt


@torch.no_grad()
def validate_textual(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.textual_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, state.pitch_prediction),
    )
    log.add_loss(
        "energy", torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction)
    )
    loss_ce, loss_dur = compute_duration_ce_loss(
        state.duration_prediction,
        state.duration_results[1].sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    return log, state.get_attention(), pred.audio, batch.audio_gt


@torch.no_grad()
def validate_sbert(batch, train):
    """Validation function for the sbert stage."""
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.sbert_prediction_single(batch)
    # 1. Get textual and acoustic embeddings
    textual_style_embedding = state.textual_style_embedding(batch.sentence_embedding)
    textual_prosody_embedding = state.textual_prosody_embedding(
        batch.sentence_embedding
    )
    acoustic_style_embedding = state.acoustic_style_embedding(batch.mel)
    acoustic_prosody_embedding = state.acoustic_prosody_embedding(batch.mel)

    log = build_loss_log(train)

    # 2. Calculate Loss
    style_loss = torch.nn.functional.l1_loss(
        textual_style_embedding, acoustic_style_embedding
    )
    prosody_loss = torch.nn.functional.l1_loss(
        textual_prosody_embedding, acoustic_prosody_embedding
    )

    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), batch.audio_gt))
    log.add_loss("sbert_style_loss", style_loss)
    log.add_loss("sbert_prosody_loss", prosody_loss)
    return log, state.get_attention(), pred.audio[0], batch.audio_gt
