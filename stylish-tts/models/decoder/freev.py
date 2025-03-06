import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import numpy as np
from munch import Munch
from librosa.filters import mel as librosa_mel_fn
import random

import math

# import torch
# from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from utils import DecoderPrediction

# import torch.nn.functional as F

mel_window = {}
inv_mel_window = {}


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    in_dataset=False,
):
    global inv_mel_window, mel_window
    device = torch.device("cpu") if in_dataset else mel.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in inv_mel_window:
        inv_basis = inv_mel_window[ps]
    else:
        if ps in mel_window:
            mel_basis, _ = mel_window[ps]
        else:
            mel_np = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            mel_window[ps] = (mel_basis.clone(), hann_window.clone())
        inv_basis = mel_basis.pinverse()
        inv_mel_window[ps] = inv_basis.clone()
    return inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value=None,
        adanorm_num_embeddings=None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


config_h = Munch(
    {
        "ASP_channel": 1025,
        "ASP_resblock_kernel_sizes": [3, 7, 11],
        "ASP_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "ASP_input_conv_kernel_size": 7,
        "ASP_output_conv_kernel_size": 7,
        "PSP_channel": 512,
        "PSP_resblock_kernel_sizes": [3, 7, 11],
        "PSP_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "PSP_input_conv_kernel_size": 7,
        "PSP_output_R_conv_kernel_size": 7,
        "PSP_output_I_conv_kernel_size": 7,
        "num_mels": 80,
        "n_fft": 2048,
        "hop_size": 300,
        "win_size": 1200,
        "sampling_rate": 24000,
        "fmin": 50,
        "fmax": 550,
    }
)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        h = config_h
        self.h = config_h
        self.ASP_num_kernels = len(h.ASP_resblock_kernel_sizes)
        self.PSP_num_kernels = len(h.PSP_resblock_kernel_sizes)

        # self.ASP_input_conv = Conv1d(
        #     h.num_mels,
        #     h.ASP_channel,
        #     h.ASP_input_conv_kernel_size,
        #     1,
        #     padding=get_padding(h.ASP_input_conv_kernel_size, 1),
        # )
        self.PSP_input_conv = Conv1d(
            h.num_mels,
            h.PSP_channel,
            h.PSP_input_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_input_conv_kernel_size, 1),
        )

        # self.ASP_output_conv = Conv1d(
        #     h.ASP_channel,
        #     h.n_fft // 2 + 1,
        #     h.ASP_output_conv_kernel_size,
        #     1,
        #     padding=get_padding(h.ASP_output_conv_kernel_size, 1),
        # )
        self.PSP_output_R_conv = Conv1d(
            512,
            h.n_fft // 2 + 1,
            h.PSP_output_R_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_output_R_conv_kernel_size, 1),
        )
        self.PSP_output_I_conv = Conv1d(
            512,
            h.n_fft // 2 + 1,
            h.PSP_output_I_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_output_I_conv_kernel_size, 1),
        )

        self.dim = 512
        self.num_layers = 8
        self.adanorm_num_embeddings = None
        self.intermediate_dim = 1536
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value = 1 / self.num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convnext2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.h.ASP_channel,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                # for _ in range(self.num_layers)
                for _ in range(1)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.final_layer_norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mel, inv_mel=None):
        if inv_mel is None:
            inv_amp = (
                inverse_mel(
                    mel,
                    self.h.n_fft,
                    self.h.num_mels,
                    self.h.sampling_rate,
                    self.h.hop_size,
                    self.h.win_size,
                    self.h.fmin,
                    self.h.fmax,
                )
                .abs()
                .clamp_min(1e-5)
            )
        else:
            inv_amp = inv_mel
        logamp = inv_amp.log()
        logamp = F.pad(logamp, pad=(0, 1), mode="replicate")
        # logamp = self.ASP_input_conv(logamp)
        for conv_block in self.convnext2:
            logamp = conv_block(logamp, cond_embedding_id=None)
        # logamp = self.final_layer_norm2(logamp.transpose(1, 2))
        # logamp = logamp.transpose(1, 2)
        # logamp = self.ASP_output_conv(logamp)

        pha = self.PSP_input_conv(mel)
        pha = self.norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        for conv_block in self.convnext:
            pha = conv_block(pha, cond_embedding_id=None)
        pha = self.final_layer_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I, R)
        pha = F.pad(pha, pad=(0, 1), mode="replicate")
        # rea is the real part of the complex number
        rea = torch.exp(logamp) * torch.cos(pha)
        # imag is the imaginary part of the complex number
        imag = torch.exp(logamp) * torch.sin(pha)

        spec = torch.complex(rea, imag)
        # spec = torch.cat((rea.unsqueeze(-1), imag.unsqueeze(-1)), -1)

        audio = torch.istft(
            spec,
            self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            window=torch.hann_window(self.h.win_size).to(mel.device),
            center=True,
        )

        # return logamp, pha, rea, imag, audio.unsqueeze(1)
        return (
            audio.unsqueeze(1),
            logamp,
            pha,
            rea,
            imag,
        )


###################################################################


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


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in=512,
        style_dim=128,
        residual_dim=64,
        dim_out=80,
        intermediate_dim=1536,
        num_layers=8,
    ):
        super().__init__()

        self.decode = nn.ModuleList()

        self.bottleneck_dim = dim_in * 2

        self.encode = nn.Sequential(
            ResBlk1d(dim_in + 2, self.bottleneck_dim, normalize=True),
            ResBlk1d(self.bottleneck_dim, self.bottleneck_dim, normalize=True),
        )

        self.decode.append(
            AdainResBlk1d(
                self.bottleneck_dim + residual_dim + 2, self.bottleneck_dim, style_dim
            )
        )
        self.decode.append(
            AdainResBlk1d(
                self.bottleneck_dim + residual_dim + 2, self.bottleneck_dim, style_dim
            )
        )
        self.decode.append(
            AdainResBlk1d(
                self.bottleneck_dim + residual_dim + 2, dim_in, style_dim, upsample=True
            )
        )
        self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))
        self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))

        self.F0_conv = nn.Sequential(
            ResBlk1d(1, residual_dim, normalize=True, downsample=True),
            weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
            nn.InstanceNorm1d(1, affine=True),
        )

        self.N_conv = nn.Sequential(
            ResBlk1d(1, residual_dim, normalize=True, downsample=True),
            weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
            nn.InstanceNorm1d(1, affine=True),
        )

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, residual_dim, kernel_size=1)),
            nn.InstanceNorm1d(residual_dim, affine=True),
        )

        self.to_out = nn.Sequential(weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0)))

        self.generator = Generator()

    def forward(self, asr, F0, N, s, pretrain=False, probing=False):
        if not pretrain:
            # if self.training:
            #    downlist = [0, 3, 7]
            #    F0_down = downlist[random.randint(0, 2)]
            #    downlist = [0, 3, 7, 15]
            #    N_down = downlist[random.randint(0, 3)]
            #    if probing:
            #        F0_down = 0
            #        N_down = 0
            #    if F0_down:
            #        F0 = (
            #            nn.functional.conv1d(
            #                F0.unsqueeze(1),
            #                torch.ones(1, 1, F0_down).to("cuda"),
            #                padding=F0_down // 2,
            #            ).squeeze(1)
            #            / F0_down
            #        )
            #    if N_down:
            #        N = (
            #            nn.functional.conv1d(
            #                N.unsqueeze(1),
            #                torch.ones(1, 1, N_down).to("cuda"),
            #                padding=N_down // 2,
            #            ).squeeze(1)
            #            / N_down
            #        )

            F0 = self.F0_conv(F0.unsqueeze(1))
            N = self.N_conv(N.unsqueeze(1))

            x = torch.cat([asr, F0, N], axis=1)
            x = self.encode(x)

            asr_res = self.asr_res(asr)

            res = True
            for block in self.decode:
                if res:
                    x = torch.cat([x, asr_res, F0, N], axis=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False
            x = self.to_out(x)
        else:
            x = asr
        audio, logamp, phase, real, imaginary = self.generator(x)
        return DecoderPrediction(
            audio=audio,
            log_amplitude=logamp,
            phase=phase,
            real=real,
            imaginary=imaginary,
        )
