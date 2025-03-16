# coding:utf-8


import math

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from config_loader import ModelConfig


from .text_aligner import TextAligner
from .plbert import PLBERT

from .discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from munch import Munch
import safetensors
from huggingface_hub import hf_hub_download

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)

import logging

logger = logging.getLogger(__name__)


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 1),
                stride=(2, 1),
                groups=dim_in,
                output_padding=(1, 0),
                padding=(1, 0),
            )
        elif self.layer_type == "half":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 3),
                stride=(2, 2),
                groups=dim_in,
                output_padding=1,
                padding=1,
            )
        else:
            raise RuntimeError(
                "Got unexpected upsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        elif self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise RuntimeError(
                "Got unexpected upsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(
        self, dim_in=48, style_dim=48, max_conv_dim=384, skip_downsamples=False
    ):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        dim_out = 0
        repeat_num = 4
        for i in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            down = "half"
            if i == repeat_num - 1 and skip_downsamples:
                down = "none"
            blocks += [ResBlk(dim_in, dim_out, downsample=down)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)

        return s


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
        dropout_p=0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.Conv1d(
                    dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1
                )
            )

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == "none":
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )

        self.prepare_projection = LinearNorm(channels, channels // 2)
        self.post_projection = LinearNorm(channels // 2, channels)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            # slstm_block=sLSTMBlockConfig(
            #     slstm=sLSTMLayerConfig(
            #         backend="cuda",
            #         num_heads=4,
            #         conv1d_kernel_size=4,
            #         bias_init="powerlaw_blockdependent",
            #     ),
            #     feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            # ),
            context_length=channels,
            num_blocks=8,
            embedding_dim=channels // 2,
            # slstm_at=[1],
        )

        self.lstm = xLSTMBlockStack(cfg)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()

        x = self.prepare_projection(x)
        x = self.lstm(x)
        x = self.post_projection(x)

        x = x.transpose(-1, -2)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )

        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


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


# class ProsodyPredictor(nn.Module):
#     def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
#         super().__init__()

#         self.text_encoder = DurationEncoder(
#             sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
#         )

#         self.lstm = nn.LSTM(
#             d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
#         )
#         self.duration_projection = LinearNorm(d_hid, max_dur)

#         self.shared = nn.LSTM(
#             d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
#         )

#         self.F0 = nn.ModuleList()
#         self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
#         self.F0.append(
#             AdainResBlk1d(
#                 d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
#             )
#         )
#         self.F0.append(
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         )

#         self.N = nn.ModuleList()
#         self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
#         self.N.append(
#             AdainResBlk1d(
#                 d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
#             )
#         )
#         self.N.append(
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         )

#         self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
#         self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

#     def forward(self, values, predict_F0N=False):
#         if predict_F0N:
#             (x, s) = values
#             x, _ = self.shared(x.transpose(-1, -2))

#             F0 = x.transpose(-1, -2)
#             for block in self.F0:
#                 F0 = block(F0, s)
#             F0 = self.F0_proj(F0)

#             N = x.transpose(-1, -2)
#             for block in self.N:
#                 N = block(N, s)
#             N = self.N_proj(N)

#             return F0.squeeze(1), N.squeeze(1)
#         else:
#             (texts, style, text_lengths, alignment, m) = values
#             d = self.text_encoder(texts, style, text_lengths, m)

#             batch_size = d.shape[0]
#             text_size = d.shape[1]

#             # predict duration
#             input_lengths = text_lengths.cpu().numpy()
#             x = nn.utils.rnn.pack_padded_sequence(
#                 d, input_lengths, batch_first=True, enforce_sorted=False
#             )

#             m = m.to(text_lengths.device).unsqueeze(1)

#             self.lstm.flatten_parameters()
#             x, _ = self.lstm(x)
#             x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

#             x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

#             x_pad[:, : x.shape[1], :] = x
#             x = x_pad.to(x.device)

#             duration = self.duration_projection(
#                 nn.functional.dropout(x, 0.5, training=self.training)
#             )

#             en = d.transpose(-1, -2) @ alignment

#             return duration.squeeze(-1), en

#     def length_to_mask(self, lengths):
#         mask = (
#             torch.arange(lengths.max())
#             .unsqueeze(0)
#             .expand(lengths.shape[0], -1)
#             .type_as(lengths)
#         )
#         mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#         return mask


class DurationEncoder(nn.Module):
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
                    # dropout=dropout,
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

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], dim=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


def build_model(model_config: ModelConfig):
    text_aligner = TextAligner(
        input_dim=model_config.n_mels,
        n_token=model_config.text_encoder.n_token,
        **(model_config.text_aligner.model_dump()),
    )
    # pitch_extractor = PitchExtractor(**(model_config.pitch_extractor.dict()))
    bert = PLBERT(
        vocab_size=model_config.text_encoder.n_token,
        **(model_config.plbert.model_dump()),
    )

    assert model_config.decoder.type in [
        "istftnet",
        # "hifigan",
        "ringformer",
        # "vocos",
        # "freev",
    ], "Decoder type unknown"

    if model_config.decoder.type == "istftnet":
        from .decoder.istftnet import Decoder

        decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            resblock_kernel_sizes=model_config.decoder.resblock_kernel_sizes,
            upsample_rates=model_config.decoder.upsample_rates,
            upsample_initial_channel=model_config.decoder.upsample_initial_channel,
            resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.decoder.gen_istft_n_fft,
            gen_istft_hop_size=model_config.decoder.gen_istft_hop_size,
            sample_rate=model_config.sample_rate,
        )
    elif model_config.decoder.type == "ringformer":
        from .decoder.ringformer import Decoder

        decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.n_mels,
            resblock_kernel_sizes=model_config.decoder.resblock_kernel_sizes,
            upsample_rates=model_config.decoder.upsample_rates,
            upsample_initial_channel=model_config.decoder.upsample_initial_channel,
            resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.decoder.gen_istft_n_fft,
            gen_istft_hop_size=model_config.decoder.gen_istft_hop_size,
            conformer_depth=model_config.decoder.depth,
            sample_rate=model_config.sample_rate,
        )
    # elif model_config.decoder.type == "vocos":
    #     from .decoder.vocos import Decoder

    #     decoder = Decoder(
    #         dim_in=model_config.decoder.hidden_dim,
    #         style_dim=model_config.style_dim,
    #         dim_out=model_config.n_mels,
    #         intermediate_dim=model_config.decoder.intermediate_dim,
    #         num_layers=model_config.decoder.num_layers,
    #         gen_istft_n_fft=model_config.decoder.gen_istft_n_fft,
    #         gen_istft_hop_size=model_config.decoder.gen_istft_hop_size,
    #     )
    # elif model_config.decoder.type == "freev":
    #     from .decoder.freev import Decoder

    #     decoder = Decoder()
    # else:
    #     from .decoder.hifigan import Decoder

    #     decoder = Decoder(
    #         dim_in=model_config.decoder.hidden_dim,
    #         style_dim=model_config.style_dim,
    #         dim_out=model_config.n_mels,
    #         resblock_kernel_sizes=model_config.decoder.resblock_kernel_sizes,
    #         upsample_rates=model_config.decoder.upsample_rates,
    #         upsample_initial_channel=model_config.decoder.upsample_initial_channel,
    #         resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
    #         upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
    #     )

    text_encoder = TextEncoder(
        channels=model_config.inter_dim,
        kernel_size=model_config.text_encoder.kernel_size,
        depth=model_config.text_encoder.n_layer,
        n_symbols=model_config.text_encoder.n_token,
    )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        nlayers=model_config.duration_predictor.n_layer,
        max_dur=model_config.duration_predictor.max_dur,
        dropout=model_config.duration_predictor.dropout,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        dropout=model_config.pitch_energy_predictor.dropout,
    )

    # predictor = ProsodyPredictor(
    #    style_dim=model_config.style_dim,
    #    d_hid=model_config.prosody_predictor.hidden_dim,
    #    nlayers=model_config.prosody_predictor.n_layer,
    #    max_dur=model_config.prosody_predictor.max_dur,
    #    dropout=model_config.prosody_predictor.dropout,
    # )

    style_encoder = StyleEncoder(
        dim_in=model_config.style_encoder.dim_in,
        style_dim=model_config.style_dim,
        max_conv_dim=model_config.style_encoder.hidden_dim,
        skip_downsamples=model_config.style_encoder.skip_downsamples,
    )
    predictor_encoder = StyleEncoder(
        dim_in=model_config.style_encoder.dim_in,
        style_dim=model_config.style_dim,
        max_conv_dim=model_config.style_encoder.hidden_dim,
        skip_downsamples=model_config.style_encoder.skip_downsamples,
    )

    nets = Munch(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, model_config.inter_dim),
        # predictor=predictor,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        decoder=decoder,
        text_encoder=text_encoder,
        # TODO Make this a config option
        # TODO Make the sbert model a config option
        textual_prosody_encoder=nn.Linear(
            384,  # model_config.embedding_encoder.dim_in,
            model_config.style_dim,
        ),
        textual_style_encoder=nn.Linear(
            384,  # model_config.embedding_encoder.dim_in,
            model_config.style_dim,
        ),
        acoustic_prosody_encoder=predictor_encoder,
        acoustic_style_encoder=style_encoder,
        # diffusion=diffusion,
        text_aligner=text_aligner,
        # pitch_extractor=pitch_extractor,
        mpd=MultiPeriodDiscriminator(),
        msd=MultiScaleSubbandCQTDiscriminator(sample_rate=model_config.sample_rate),
        # msd=MultiResolutionDiscriminator(),
        # slm discriminator head
        # wd=WavLMDiscriminator(
        #    model_config.slm.hidden,
        #    model_config.slm.nlayers,
        #    model_config.slm.initial_channel,
        # ),
    )

    return nets  # , kdiffusion


def load_defaults(train, model):
    with train.accelerator.main_process_first():
        # Load pretrained text_aligner
        params = safetensors.torch.load_file(
            hf_hub_download(
                repo_id="stylish-tts/text_aligner", filename="text_aligner.safetensors"
            )
        )
        model.text_aligner.load_state_dict(params)

        # Load pretrained pitch_extractor
        # params = safetensors.torch.load_file(
        # hf_hub_download(
        # repo_id="stylish-tts/pitch_extractor",
        # filename="pitch_extractor.safetensors",
        # )
        # )
        # model.pitch_extractor.load_state_dict(params)

        # Load pretrained PLBERT
        params = safetensors.torch.load_file(
            hf_hub_download(repo_id="stylish-tts/plbert", filename="plbert.safetensors")
        )
        model.bert.load_state_dict(params, strict=False)
