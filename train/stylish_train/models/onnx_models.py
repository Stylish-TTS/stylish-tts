import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce

# import train_context
from utils import length_to_mask, log_norm
from models.models import build_model
import torch
import torch.nn as nn


from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSTFT(nn.Module):
    """
    STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

    - forward STFT => Real-part conv1d + Imag-part conv1d
    - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d + sum
    - avoids F.unfold, so easier to export to ONNX
    - uses replicate or constant padding for 'center=True' to approximate 'reflect'
      (reflect is not supported for dynamic shapes in ONNX)
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",  # or 'constant'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window
        assert window == "hann", window
        window_tensor = torch.hann_window(
            win_length, periodic=True, dtype=torch.float32
        )
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)

        # Precompute forward DFT (real, imag)
        # PyTorch stft uses e^{-j 2 pi k n / N} => real=cos(...), imag=-sin(...)
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft  # shape (freq_bins, n_fft)
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)  # note negative sign

        # Combine window and dft => shape (freq_bins, filter_length)
        # We'll make 2 conv weight tensors of shape (freq_bins, 1, filter_length).
        forward_window = window_tensor.numpy()  # shape (n_fft,)
        forward_real = dft_real * forward_window  # (freq_bins, n_fft)
        forward_imag = dft_imag * forward_window

        # Convert to PyTorch
        forward_real_torch = torch.from_numpy(forward_real).float()
        forward_imag_torch = torch.from_numpy(forward_imag).float()

        # Register as Conv1d weight => (out_channels, in_channels, kernel_size)
        # out_channels = freq_bins, in_channels=1, kernel_size=n_fft
        self.register_buffer("weight_forward_real", forward_real_torch.unsqueeze(1))
        self.register_buffer("weight_forward_imag", forward_imag_torch.unsqueeze(1))

        # Precompute inverse DFT
        # Real iFFT formula => scale = 1/n_fft, doubling for bins 1..freq_bins-2 if n_fft even, etc.
        # For simplicity, we won't do the "DC/nyquist not doubled" approach here.
        # If you want perfect real iSTFT, you can add that logic.
        # This version just yields good approximate reconstruction with Hann + typical overlap.
        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft  # shape (n_fft, freq_bins)
        idft_cos = np.cos(angle_t).T  # => (freq_bins, n_fft)
        idft_sin = np.sin(angle_t).T  # => (freq_bins, n_fft)

        # Multiply by window again for typical overlap-add
        # We also incorporate the scale factor 1/n_fft
        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window  # (freq_bins, n_fft)
        backward_imag = idft_sin * inv_window

        # We'll implement iSTFT as real+imag conv_transpose with stride=hop.
        self.register_buffer(
            "weight_backward_real", torch.from_numpy(backward_real).float().unsqueeze(1)
        )
        self.register_buffer(
            "weight_backward_imag", torch.from_numpy(backward_imag).float().unsqueeze(1)
        )

    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT => returns magnitude, phase
        Output shape => (batch, freq_bins, frames)
        """
        # waveform shape => (B, T).  conv1d expects (B, 1, T).
        # Optional center pad
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)  # => (B, 1, T)
        # Convolution to get real part => shape (B, freq_bins, frames)
        real_out = F.conv1d(
            x,
            self.weight_forward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # Imag part
        imag_out = F.conv1d(
            x,
            self.weight_forward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )

        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        # In this case, PyTorch returns pi, ONNX returns -pi
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        """
        Inverse STFT => returns waveform shape (B, T).
        """
        # magnitude, phase => (B, freq_bins, frames)
        # Re-create real/imag => shape (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # conv_transpose wants shape (B, freq_bins, frames). We'll treat "frames" as time dimension
        # so we do (B, freq_bins, frames) => (B, freq_bins, frames)
        # But PyTorch conv_transpose1d expects (B, in_channels, input_length)
        real_part = real_part  # (B, freq_bins, frames)
        imag_part = imag_part

        # real iSTFT => convolve with "backward_real", "backward_imag", and sum
        # We'll do 2 conv_transpose calls, each giving (B, 1, time),
        # then add them => (B, 1, time).
        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # sum => (B, 1, time)
        waveform = real_rec - imag_rec  # typical real iFFT has minus for imaginary part

        # If we used "center=True" in forward, we should remove pad
        if self.center:
            pad_len = self.n_fft // 2
            # Because of transposed convolution, total length might have extra samples
            # We remove `pad_len` from start & end if possible
            waveform = waveform[..., pad_len:-pad_len]

        # If a specific length is desired, clamp
        if length is not None:
            waveform = waveform[..., :length]

        # shape => (B, T)
        return waveform

    def forward(self, x: torch.Tensor):
        """
        Full STFT -> iSTFT pass: returns time-domain reconstruction.
        Same interface as your original code.
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])


class Stylish(nn.Module):
    def __init__(
        self,
        text_encoder,
        textual_style_encoder,
        textual_prosody_encoder,
        bert,
        bert_encoder,
        duration_predictor,
        pitch_energy_predictor,
        decoder,
        generator,
        prosody_lstm=None,
        device="cuda",
        **kwargs
    ):
        super(Stylish, self).__init__()

        for model in [
            text_encoder,
            textual_style_encoder,
            textual_prosody_encoder,
            bert,
            bert_encoder,
            duration_predictor,
            pitch_energy_predictor,
            decoder,
            generator,
            prosody_lstm,
        ]:
            if model is None:
                continue
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.text_encoder = text_encoder
        self.textual_style_encoder = textual_style_encoder
        self.textual_prosody_encoder = textual_prosody_encoder
        self.bert = bert
        self.bert_encoder = bert_encoder
        self.duration_predictor = duration_predictor
        self.pitch_energy_predictor = pitch_energy_predictor
        self.decoder = decoder
        self.generator = generator
        self.prosody_lstm = prosody_lstm
        self.plbert_enabled = self.prosody_lstm is None

    def duration_encoding(self, texts, text_mask):
        if self.plbert_enabled:
            plbert_embedding = self.model.bert(texts, attention_mask=(~text_mask).int())
            return self.model.bert_encoder(plbert_embedding).transpose(-1, -2)
        else:
            return self.model.text_encoder.infer(texts, external_lstm=self.prosody_lstm)

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
    ):
        mel, _ = self.decoder(
            text_encoding @ duration, pitch, energy, style, probing=False
        )
        prediction = self.generator(mel=mel, style=style, pitch=pitch, energy=energy)
        return prediction

    def duration_predict(self, duration_encoding, prosody_embedding):
        d = self.duration_predictor.text_encoder.infer(
            duration_encoding, prosody_embedding
        )
        x, _ = self.duration_predictor.lstm(d)
        duration = self.duration_predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)

        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(duration_encoding.shape[2], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (duration_encoding.shape[2], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        prosody = d.permute(0, 2, 1) @ pred_aln_trg
        return pred_aln_trg, prosody

    def forward(self, texts, text_mask, sentence_embedding):
        text_encoding = self.text_encoder.infer(texts)
        style_embedding = self.textual_style_encoder(sentence_embedding)
        prosody_embedding = self.textual_prosody_encoder(sentence_embedding)
        duration_encoding = self.duration_encoding(texts, text_mask)
        duration_prediction, prosody = self.duration_predict(
            duration_encoding,
            prosody_embedding,
        )
        pitch_prediction, energy_prediction = self.pitch_energy_predictor(
            prosody, prosody_embedding
        )

        prediction = self.decoding_single(
            text_encoding,
            duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
        )
        return prediction.audio.squeeze()
