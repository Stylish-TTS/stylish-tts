import torch
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log, combine_logs
from losses import freev_loss, magphase_loss


def train_pre_acoustic(batch, model, train) -> LossLog:
    split_count = 1
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        decoding = state.acoustic_prediction(batch, split=split_count)
        train.stage.optimizer.zero_grad()
        loglist = []
        for pred, audio_gt_slice, _, _ in decoding:
            log = build_loss_log(train)
            log.add_loss(
                "mel",
                train.stft_loss(pred.audio.squeeze(1), audio_gt_slice) / split_count,
            )
            train.accelerator.backward(log.total(), retain_graph=True)
            loglist.append(log)
    return combine_logs(loglist).detach()


def train_acoustic(batch, model, train) -> LossLog:
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch)
        train.stage.optimizer.zero_grad()
        d_loss = train.discriminator_loss(
            batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        ).mean()
        train.accelerator.backward(d_loss)
        train.stage.optimizer.step("msd")
        train.stage.optimizer.step("mpd")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        log.add_loss(
            "mel",
            train.stft_loss(pred.audio.squeeze(1), batch.audio_gt),
        )
        log.add_loss(
            "gen",
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

        loss_s2s = 0
        for durpred, text, length in zip(state.s2s_pred, batch.text, batch.text_length):
            loss_s2s += torch.nn.functional.cross_entropy(
                durpred[:length], text[:length]
            )
        loss_s2s /= batch.text.size(0)
        log.add_loss("s2s", loss_s2s)

        log.add_loss(
            "mono", torch.nn.functional.l1_loss(*(state.duration_results)) * 10
        )

        freev_loss(log, batch, pred, train)
        train.accelerator.backward(log.total())
        log.add_loss("discriminator", d_loss)

    return log.detach()
