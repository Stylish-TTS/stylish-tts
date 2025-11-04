import math
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from .text_encoder import MultiHeadAttention, TextEncoder
from .prosody_encoder import ProsodyEncoder
from stylish_tts.train.utils import length_to_mask
from .ada_norm import AdaptiveLayerNorm, AdaptiveDecoderBlock
from .conv_next import AdaptiveConvNeXtBlock


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        inter_dim,
        text_config,
        duration_config,
        pitch_energy_config,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(
            inter_dim=inter_dim,
            config=text_config,
        )
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=3,
            dropout=0.2,
        )

        dropout = pitch_energy_config.dropout

        cross_channels = inter_dim + style_dim
        self.query_norm = AdaptiveLayerNorm(style_dim, cross_channels)
        self.key_norm = AdaptiveLayerNorm(style_dim, cross_channels)
        self.cross_attention = MultiHeadAttention(
            channels=cross_channels,
            out_channels=cross_channels,
            n_heads=8,
            p_dropout=dropout,
        )
        self.cross_window = 5

        self.cross_post = torch.nn.Sequential(
            weight_norm(
                torch.nn.Conv1d(
                    cross_channels,
                    cross_channels,
                    kernel_size=5,
                    padding=2,
                    groups=cross_channels,
                )
            ),
            torch.nn.SiLU(),
            weight_norm(torch.nn.Conv1d(cross_channels, cross_channels, kernel_size=1)),
        )

        self.F0 = torch.nn.ModuleList(
            [
                AdaptiveDecoderBlock(
                    inter_dim + style_dim,
                    inter_dim + style_dim,
                    style_dim,
                    dropout_p=dropout,
                    kernel_size=13,
                )
                for _ in range(4)
            ]
        )

        self.N = torch.nn.ModuleList(
            [
                AdaptiveDecoderBlock(
                    inter_dim + style_dim,
                    inter_dim + style_dim,
                    style_dim,
                    dropout_p=dropout,
                    kernel_size=13,
                )
                for _ in range(4)
            ]
        )

        self.F0_post = torch.nn.ModuleList(
            [
                AdaptiveConvNeXtBlock(
                    inter_dim + style_dim,
                    (inter_dim + style_dim) * 3,
                    style_dim,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.N_post = torch.nn.ModuleList(
            [
                AdaptiveConvNeXtBlock(
                    inter_dim + style_dim,
                    (inter_dim + style_dim) * 3,
                    style_dim,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )

        self.F0_proj = torch.nn.Conv1d(inter_dim + style_dim, 1, 1, 1, 0)
        self.N_proj = torch.nn.Conv1d(inter_dim + style_dim, 1, 1, 1, 0)
        self.voiced_proj = torch.nn.Conv1d(inter_dim + style_dim, 1, 1, 1, 0)
        self.sigmoid = torch.nn.Sigmoid()

    def compute_cross(self, text_encoding, alignment, style, text_mask):
        """
        d_tok: [B, T, C] token states (style-conditioned)
        alignment: [B, T, F] monotonic alignment
        style: [B, S]
        text_mask: [B, T] True at padding
        Returns: en [B, C, F]
        """
        base = torch.matmul(text_encoding.transpose(1, 2), alignment)
        query = base.transpose(1, 2)
        query = self.query_norm(query, style).transpose(1, 2)
        key = text_encoding  # [B, T, C]
        key = self.key_norm(key, style).transpose(1, 2)

        attention_mask = build_monotonic_band_mask(
            alignment, text_mask, self.cross_window
        )  # [B, 1, F, T]
        attention = self.cross_attention(query, key, attn_mask=attention_mask)
        attention = self.cross_post(attention)
        return (base + attention) / math.sqrt(2.0)

    def forward(self, texts, text_lengths, alignment, style):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        mask = length_to_mask(text_lengths, text_encoding.shape[2]).to(
            text_lengths.device
        )
        prosody = self.prosody_encoder(text_encoding, style, text_lengths)
        x = self.compute_cross(prosody, alignment, style, mask)

        F0 = x
        for block in self.F0:
            F0 = block(F0, style)
        F0 = torch.nn.functional.interpolate(F0, scale_factor=4, mode="linear")
        for block in self.F0_post:
            F0 = block(F0, style)
        voiced = self.voiced_proj(F0)
        voiced = self.sigmoid(voiced)
        F0 = self.F0_proj(F0)

        N = x
        for block in self.N:
            N = block(N, style)
        N = torch.nn.functional.interpolate(N, scale_factor=4, mode="linear")
        for block in self.N_post:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1), voiced.squeeze(1)


def build_monotonic_band_mask(alignment, text_mask, window):
    """
    alignment: [B, T, F] (monotonic hard/soft align)
    text_mask: [B, T] True at padding
    Returns attn_mask: [B, 1, F, T] True where attention is NOT allowed.
    """
    with torch.no_grad():
        B, T, F = alignment.shape
        tau = alignment.argmax(dim=1)
        t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
        tau_exp = tau.unsqueeze(-1)
        band = (t_idx >= (tau_exp - window)) & (t_idx <= (tau_exp + window))

        band_mask = ~band

        # Also mask padded tokens
        key_pad = text_mask.unsqueeze(1).expand(B, F, T)
        full_mask = band_mask | key_pad
        return full_mask.unsqueeze(1)
