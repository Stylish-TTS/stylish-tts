import math
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from .text_encoder import MultiHeadAttention, TextEncoder
from .prosody_encoder import ProsodyEncoder
from stylish_tts.train.utils import length_to_mask
from .ada_norm import AdaptiveLayerNorm, AdaptiveDecoderBlock
from .conv_next import AdaptiveConvNeXtBlock


import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import math


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        torch._check(x.shape[2] > 1)
        return (1 + gamma) * self.norm(x) + beta


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        inter_dim,
        coarse_multiplier,
        text_config,
        duration_config,
        pitch_energy_config,
    ):
        super().__init__()
        self.coarse_multiplier = coarse_multiplier
        self.text_encoder = TextEncoder(
            inter_dim=inter_dim,
            config=text_config,
        )
        dropout = 0.2
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=3,
            dropout=0.2,
        )

        # self.shared = nn.LSTM(
        #     d_hid + style_dim, d_hid, 1, batch_first=True, bidirectional=True
        # )

        repeat = int(math.log2(self.coarse_multiplier))
        assert (
            2**repeat == self.coarse_multiplier
        ), "coarse_multiplier must be a power of 2"
        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(
                    inter_dim + style_dim, inter_dim, style_dim, dropout_p=dropout
                ),
                *[
                    AdainResBlk1d(
                        inter_dim // (2**i),
                        inter_dim // (2 ** (i + 1)),
                        style_dim,
                        dropout_p=dropout,
                        upsample=True,
                    )
                    for i in range(repeat)
                ],
                AdainResBlk1d(
                    inter_dim // (2**repeat),
                    inter_dim // (2**repeat),
                    style_dim,
                    dropout_p=dropout,
                ),
            ]
        )

        self.N = nn.ModuleList(
            [
                AdainResBlk1d(
                    inter_dim + style_dim, inter_dim, style_dim, dropout_p=dropout
                ),
                *[
                    AdainResBlk1d(
                        inter_dim // (2**i),
                        inter_dim // (2 ** (i + 1)),
                        style_dim,
                        dropout_p=dropout,
                        upsample=True,
                    )
                    for i in range(repeat)
                ],
                AdainResBlk1d(
                    inter_dim // (2**repeat),
                    inter_dim // (2**repeat),
                    style_dim,
                    dropout_p=dropout,
                ),
            ]
        )

        self.F0_proj = nn.Conv1d(inter_dim // (2**repeat), 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(inter_dim // (2**repeat), 1, 1, 1, 0)

        dropout = pitch_energy_config.dropout

    def forward(self, texts, text_lengths, alignment, style):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        mask = length_to_mask(text_lengths, text_encoding.shape[2]).to(
            text_lengths.device
        )
        prosody = self.prosody_encoder(text_encoding, style, text_lengths)  # , mask)

        x = prosody.transpose(1, 2) @ alignment
        x = x.transpose(-1, -2)
        # x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

        # x = self.compute_cross(prosody, alignment, style, mask)
        # x = prosody.transpose(1, 2)

        # F0 = x
        # for block in self.F0:
        #     F0 = block(F0, style)
        # voiced = F0
        # for block in self.voiced_post:
        #     voiced = block(voiced, style)

        # for block in self.F0_post:
        #     F0 = block(F0, style)
        # F0 = torch.nn.functional.dropout(F0, 0.5)
        # F0 = self.F0_proj(F0)
        # F0 = torch.matmul(F0, alignment)
        # F0 = torch.nn.functional.interpolate(
        #     F0, scale_factor=self.coarse_multiplier, mode="linear"
        # )

        # voiced = x
        # for block in self.voiced:
        #     voiced = block(voiced, style)
        # voiced = torch.nn.functional.interpolate(
        #     voiced, scale_factor=self.coarse_multiplier, mode="linear"
        # )
        # for block in self.voiced_post:
        #     voiced = block(voiced, style)
        # voiced = torch.nn.functional.dropout(voiced, 0.5)
        # voiced = self.voiced_proj(voiced)
        # voiced = self.sigmoid(voiced)

        # N = x
        # for block in self.N:
        #     N = block(N, style)
        # for block in self.N_post:
        #     N = block(N, style)
        # N = torch.nn.functional.dropout(N, 0.5)
        # N = self.N_proj(N)
        # N = torch.matmul(N, alignment)
        # N = torch.nn.functional.interpolate(
        #     N, scale_factor=self.coarse_multiplier, mode="linear"
        # )

        # return F0.squeeze(1), N.squeeze(1), voiced.squeeze(1)


# def build_monotonic_band_mask(alignment, text_mask, window):
#     """
#     alignment: [B, T, F] (monotonic hard/soft align)
#     text_mask: [B, T] True at padding
#     Returns attn_mask: [B, 1, F, T] True where attention is NOT allowed.
#     """
#     with torch.no_grad():
#         B, T, F = alignment.shape
#         tau = alignment.argmax(dim=1)
#         t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
#         tau_exp = tau.unsqueeze(-1)
#         band = (t_idx >= (tau_exp - window)) & (t_idx <= (tau_exp + window))
#
#         band_mask = ~band
#
#         # Also mask padded tokens
#         key_pad = text_mask.unsqueeze(1).expand(B, F, T)
#         full_mask = band_mask | key_pad
#         return full_mask.unsqueeze(1)


class ProsodyEncoderOld(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)

        return x.transpose(-1, -2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
