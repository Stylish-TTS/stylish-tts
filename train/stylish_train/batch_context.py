import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
from monotonic_align import mask_from_lens
import train_context
from config_loader import Config
from utils import length_to_mask, log_norm, maximum_path


class BatchContext:
    def __init__(
        self,
        *,
        train: train_context.TrainContext,
        model,
        text_length: Optional[torch.Tensor],
    ):
        self.train: train_context.TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.text_mask = None
        if text_length is not None:
            self.text_mask = length_to_mask(text_length).to(self.config.training.device)
        self.duration_results = None
        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

    def text_encoding(self, texts: torch.Tensor, alignment: torch.Tensor):
        text_spread = texts.unsqueeze(1).float() @ alignment
        text_spread = F.interpolate(text_spread, scale_factor=2, mode="nearest-exact")
        text_spread = text_spread.squeeze(1).int()
        embedding = self.model.text_encoder(text_spread)
        embedding = rearrange(embedding, "b t f -> b f t")
        return embedding
        # return self.model.text_encoder(texts, text_lengths, self.text_mask)

    def bert_encoding(self, texts: torch.Tensor):
        mask = (~self.text_mask).int()
        bert_encoding = self.model.bert(texts, attention_mask=mask)
        text_encoding = self.model.bert_encoder(bert_encoding)
        return text_encoding.transpose(-1, -2)

    def acoustic_duration(
        self,
        batch,
        # mels: torch.Tensor,
        # mel_lengths: torch.Tensor,
        # texts: torch.Tensor,
        # text_lengths: torch.Tensor,
        # apply_attention_mask: bool = False,
        # use_random_choice: bool = False,
    ) -> torch.Tensor:
        """
        Computes the duration used for training using a text aligner on
        the combined ground truth audio and text.
        Returns:
          - duration: Duration attention vector
        """
        duration = batch.alignment
        self.attention = duration[0]
        self.duration_results = (duration, duration)
        return duration

    def get_attention(self):
        return self.attention

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def calculate_pitch(self, batch, prediction=None):
        if prediction is None:
            prediction = batch.pitch
        # mask = batch.voiced.unsqueeze(1)
        # mask = mask @ self.duration_results[1]
        # mask = mask.squeeze(1).repeat_interleave(repeats=2, dim=1)
        # return prediction * mask
        return prediction

    def acoustic_style_embedding(self, mels: torch.Tensor, mel_lengths: torch.Tensor):
        return self.model.acoustic_style_encoder(mels.unsqueeze(1), mel_lengths)

    def acoustic_prosody_embedding(self, mels: torch.Tensor, mel_lengths: torch.Tensor):
        return self.model.acoustic_prosody_encoder(mels.unsqueeze(1), mel_lengths)

    def textual_style_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_style_encoder(sentence_embedding)

    def textual_prosody_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_prosody_encoder(sentence_embedding)

    def decoding(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        audio_gt,
        split=1,
        probing=False,
    ):
        if split == 1 or text_encoding.shape[0] != 1:
            prediction = self.model.decoder(
                text_encoding @ duration, pitch, energy, style, probing=probing
            )
            yield (prediction, audio_gt, 0, energy.shape[-1])
        else:
            hop_length = train.model_config.hop_length
            text_hop = text_encoding.shape[-1] // split
            text_start = 0
            text_end = text_hop + text_encoding.shape[-1] % split
            mel_start = 0
            mel_end = 0
            for i in range(split):
                mel_hop = int(duration[:, text_start:text_end, :].sum().item())
                mel_start = mel_end
                mel_end = mel_start + mel_hop

                text_slice = text_encoding[:, :, text_start:text_end]
                duration_slice = duration[:, text_start:text_end, mel_start:mel_end]
                pitch_slice = pitch[:, mel_start * 2 : mel_end * 2]
                energy_slice = energy[:, mel_start * 2 : mel_end * 2]
                audio_gt_slice = audio_gt[
                    :, mel_start * hop_length * 2 : mel_end * hop_length * 2
                ].detach()
                prediction = self.train.model.decoder(
                    text_slice @ duration_slice,
                    pitch_slice,
                    energy_slice,
                    style,
                    probing=probing,
                )
                yield (prediction, audio_gt_slice, mel_start, mel_end)
                text_start += text_hop
                text_end += text_hop

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        probing=False,
    ):
        # mel = self.model.decoder(
        #     text_encoding @ duration, pitch, energy, style, probing=probing
        # )
        mel = self.model.decoder(text_encoding, pitch, energy, style, probing=probing)
        prediction = self.model.generator(
            mel=mel, style=style, pitch=pitch, energy=energy
        )
        # prediction = self.model.decoder(
        #     text_encoding @ duration, pitch, energy, style, probing=probing
        # )
        return prediction

    def mel_reconstruction(self, batch, probing=False):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        duration = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=True,
            # use_random_choice=False,
        )
        energy = self.acoustic_energy(batch.mel)
        style_embedding = self.acoustic_style_embedding(batch.mel, batch.mel_length)
        pitch = self.calculate_pitch(batch).detach()
        return self.model.decoder(
            text_encoding, pitch, energy, style_embedding, probing=probing
        )

    def audio_reconstruction(self, batch):
        mel = batch.mel
        energy = self.acoustic_energy(batch.mel).detach()
        style_embedding = self.acoustic_style_embedding(batch.mel, batch.mel_length)
        pitch = self.calculate_pitch(batch).detach()
        gt = batch.audio_gt
        # if mel.shape[2] > 200:
        #     hop = self.train.model_config.hop_length
        #     mel_start = random.randint(0, mel.shape[2] - 200)
        #     mel_end = mel_start + 200
        #     mel = mel[:, :, mel_start:mel_end].detach()
        #     energy = energy[:, mel_start:mel_end].detach()
        #     pitch = pitch[:, mel_start:mel_end].detach()
        #     gt = gt[:, mel_start*hop:mel_end*hop].detach()
        prediction = self.model.generator(
            mel=mel, style=style_embedding, pitch=pitch, energy=energy
        )
        return prediction, gt

    def acoustic_prediction(self, batch, split=1):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        duration = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=True,
            # use_random_choice=True,
        )
        energy = self.acoustic_energy(batch.mel)
        style_embedding = self.acoustic_style_embedding(batch.mel, batch.mel_length)
        pitch = self.calculate_pitch(batch).detach()
        prediction = self.decoding(
            text_encoding,
            duration,
            pitch,
            energy,
            style_embedding,
            batch.audio_gt,
            split=split,
        )
        return prediction

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        duration = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=True,
            # use_random_choice=use_random_mono,
        )
        energy = self.acoustic_energy(batch.mel)
        style_embedding = self.acoustic_style_embedding(batch.mel, batch.mel_length)
        pitch = self.calculate_pitch(batch).detach()
        prediction = self.decoding_single(
            text_encoding,
            duration,
            pitch,
            energy,
            style_embedding,
        )
        return prediction

    def textual_prediction_single(self, batch):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        duration = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=False,
            # use_random_choice=False,
        )
        style_embedding = self.acoustic_style_embedding(batch.mel, batch.mel_length)
        prosody_embedding = self.acoustic_prosody_embedding(batch.mel, batch.mel_length)
        if "plbert" in self.model:
            plbert_embedding = self.model.bert(
                batch.text, attention_mask=(~self.text_mask).int()
            )
            duration_encoding = self.model.bert_encoder(plbert_embedding).transpose(-1, -2)
        else:
            duration_encoding = text_encoding
        self.duration_prediction, prosody = self.model.duration_predictor(
            duration_encoding,
            prosody_embedding,
            batch.text_length,
            self.duration_results[1],
            self.text_mask,
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(prosody, prosody_embedding)
        )
        pitch = self.calculate_pitch(batch, self.pitch_prediction)
        prediction = self.decoding_single(
            text_encoding,
            duration,
            pitch,
            self.energy_prediction,
            style_embedding,
        )
        return prediction

    def sbert_prediction_single(self, batch):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        duration = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=False,
            # use_random_choice=False,
        )
        style_embedding = self.textual_style_embedding(batch.sentence_embedding)
        prosody_embedding = self.textual_prosody_embedding(batch.sentence_embedding)
        if "plbert" in self.model:
            plbert_embedding = self.model.bert(
                batch.text, attention_mask=(~self.text_mask).int()
            )
            duration_encoding = self.model.bert_encoder(plbert_embedding).transpose(-1, -2)
        else:
            duration_encoding = text_encoding
        self.duration_prediction, prosody = self.model.duration_predictor(
            duration_encoding,
            prosody_embedding,
            batch.text_length,
            self.duration_results[1],
            self.text_mask,
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(prosody, prosody_embedding)
        )
        prediction = self.decoding_single(
            text_encoding,
            duration,
            self.pitch_prediction,
            self.energy_prediction,
            style_embedding,
        )
        return prediction

    def textual_bootstrap_prediction(self, batch):
        text_encoding = self.text_encoding(batch.text, batch.alignment)
        _ = self.acoustic_duration(
            batch,
            # batch.align_mel,
            # batch.mel_length,
            # batch.text,
            # batch.text_length,
            # apply_attention_mask=False,
            # use_random_choice=False,
        )
        prosody_embedding = self.acoustic_prosody_embedding(batch.mel)
        if "plbert" in self.model:
            plbert_embedding = self.model.bert(
                batch.text, attention_mask=(~self.text_mask).int()
            )
            duration_encoding = self.model.bert_encoder(plbert_embedding).transpose(-1, -2)
        else:
            duration_encoding = text_encoding
        self.duration_prediction, prosody = self.model.duration_predictor(
            duration_encoding,
            prosody_embedding,
            batch.text_length,
            self.duration_results[1],
            self.text_mask,
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(prosody, prosody_embedding)
        )


def soft_alignment(pred, phonemes, mask):
    """
    Args:
        pred (b t k): Predictions of k (+ blank) tokens at time frame t
        phonemes (b p): Target sequence of phonemes
        mask (b p): Mask for target sequence
    Returns:
        (b t p): Phoneme predictions for each time frame t
    """
    mask = rearrange(mask, "b p -> b 1 p")
    # Convert to <blank>, <phoneme>, <blank> ...
    # blank_id = pred.shape[2] - 1
    # blanks = torch.full_like(phonemes, blank_id)
    # ph_blank = rearrange([phonemes, blanks], "n b p -> b (p n)")
    # ph_blank = F.pad(ph_blank, (0, 1), value=blank_id)
    # ph_blank = rearrange(ph_blank, "b p -> b 1 p")
    ph_blank = rearrange(phonemes, "b p -> b 1 p")
    # pred = pred.exp()
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
    result = result * ~mask
    return result
