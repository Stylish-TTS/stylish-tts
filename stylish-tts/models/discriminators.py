import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from munch import Munch
from typing import List, Optional, Tuple

import torchaudio
from nnAudio import features
from einops import rearrange
from .norm2d import NormConv2d
from .common import get_padding

from torch.nn import Conv2d
from torchaudio.transforms import Spectrogram


# LRELU_SLOPE = 0.1
#
#
# def get_2d_padding(kernel_size, dilation=(1, 1)):
# return (
# ((kernel_size[0] - 1) * dilation[0]) // 2,
# ((kernel_size[1] - 1) * dilation[1]) // 2,
# )
#
#
# class DiscriminatorCQT(nn.Module):
# def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
# super(DiscriminatorCQT, self).__init__()
# self.cfg = cfg
#
# self.filters = cfg.filters
# self.max_filters = cfg.max_filters
# self.filters_scale = cfg.filters_scale
# self.kernel_size = (3, 9)
# self.dilations = cfg.dilations
# self.stride = (1, 2)
#
# self.in_channels = cfg.in_channels
# self.out_channels = cfg.out_channels
# self.fs = cfg.sampling_rate
# self.hop_length = hop_length
# self.n_octaves = n_octaves
# self.bins_per_octave = bins_per_octave
#
# self.cqt_transform = features.cqt.CQT2010v2(
# sr=self.fs * 2,
# hop_length=self.hop_length,
# n_bins=self.bins_per_octave * self.n_octaves,
# bins_per_octave=self.bins_per_octave,
# output_format="Complex",
# pad_mode="constant",
# )
#
# self.conv_pres = nn.ModuleList()
# for i in range(self.n_octaves):
# self.conv_pres.append(
# NormConv2d(
# self.in_channels * 2,
# self.in_channels * 2,
# kernel_size=self.kernel_size,
# padding=get_2d_padding(self.kernel_size),
# )
# )
#
# self.convs = nn.ModuleList()
#
# self.convs.append(
# NormConv2d(
# self.in_channels * 2,
# self.filters,
# kernel_size=self.kernel_size,
# padding=get_2d_padding(self.kernel_size),
# )
# )
#
# in_chs = min(self.filters_scale * self.filters, self.max_filters)
# for i, dilation in enumerate(self.dilations):
# out_chs = min(
# (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
# )
# self.convs.append(
# NormConv2d(
# in_chs,
# out_chs,
# kernel_size=self.kernel_size,
# stride=self.stride,
# dilation=(dilation, 1),
# padding=get_2d_padding(self.kernel_size, (dilation, 1)),
# norm="weight_norm",
# )
# )
# in_chs = out_chs
# out_chs = min(
# (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
# self.max_filters,
# )
# self.convs.append(
# NormConv2d(
# in_chs,
# out_chs,
# kernel_size=(self.kernel_size[0], self.kernel_size[0]),
# padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
# norm="weight_norm",
# )
# )
#
# self.conv_post = NormConv2d(
# out_chs,
# self.out_channels,
# kernel_size=(self.kernel_size[0], self.kernel_size[0]),
# padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
# norm="weight_norm",
# )
#
# self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE)
# self.resample = torchaudio.transforms.Resample(
# orig_freq=self.fs, new_freq=self.fs * 2
# )
#
# def forward(self, x):
# fmap = []
#
# x = self.resample(x)
#
# z = self.cqt_transform(x)
#
# z_amplitude = z[:, :, :, 0].unsqueeze(1)
# z_phase = z[:, :, :, 1].unsqueeze(1)
#
# z = torch.cat([z_amplitude, z_phase], dim=1)
# z = rearrange(z, "b c w t -> b c t w")
#
# latent_z = []
# for i in range(self.n_octaves):
# latent_z.append(
# self.conv_pres[i](
# z[
# :,
# :,
# :,
# i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
# ]
# )
# )
# latent_z = torch.cat(latent_z, dim=-1)
#
# for i, l in enumerate(self.convs):
# latent_z = l(latent_z)
#
# latent_z = self.activation(latent_z)
# fmap.append(latent_z)
#
# latent_z = self.conv_post(latent_z)
#
# return latent_z, fmap
#
#
# class MultiScaleSubbandCQTDiscriminator(nn.Module):
# def __init__(self, *, sample_rate):
# super(MultiScaleSubbandCQTDiscriminator, self).__init__()
# multiscale_subband_cfg = {
# "hop_lengths": [512, 256, 256],
# "sampling_rate": sample_rate,
# "filters": 32,
# "max_filters": 1024,
# "filters_scale": 1,
# "dilations": [1, 2, 4],
# "in_channels": 1,
# "out_channels": 1,
# "n_octaves": [9, 9, 9],
# "bins_per_octaves": [24, 36, 48],
# }
# cfg = Munch(multiscale_subband_cfg)
# self.cfg = cfg
# self.discriminators = nn.ModuleList(
# [
# DiscriminatorCQT(
# cfg,
# hop_length=cfg.hop_lengths[i],
# n_octaves=cfg.n_octaves[i],
# bins_per_octave=cfg.bins_per_octaves[i],
# )
# for i in range(len(cfg.hop_lengths))
# ]
# )
#
# def forward(self, y, y_hat):
# y_d_rs = []
# y_d_gs = []
# fmap_rs = []
# fmap_gs = []
#
# for disc in self.discriminators:
# y_d_r, fmap_r = disc(y)
# y_d_g, fmap_g = disc(y_hat)
# y_d_rs.append(y_d_r)
# fmap_rs.append(fmap_r)
# y_d_gs.append(y_d_g)
# fmap_gs.append(fmap_g)
#
# return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# class DiscriminatorP(torch.nn.Module):
# def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
# super(DiscriminatorP, self).__init__()
# self.period = period
# norm_f = spectral_norm if use_spectral_norm else weight_norm
# self.convs = nn.ModuleList(
# [
# norm_f(
# Conv2d(
# 1,
# 32,
# (kernel_size, 1),
# (stride, 1),
# padding=(get_padding(5, 1), 0),
# )
# ),
# norm_f(
# Conv2d(
# 32,
# 128,
# (kernel_size, 1),
# (stride, 1),
# padding=(get_padding(5, 1), 0),
# )
# ),
# norm_f(
# Conv2d(
# 128,
# 512,
# (kernel_size, 1),
# (stride, 1),
# padding=(get_padding(5, 1), 0),
# )
# ),
# norm_f(
# Conv2d(
# 512,
# 1024,
# (kernel_size, 1),
# (stride, 1),
# padding=(get_padding(5, 1), 0),
# )
# ),
# norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
# ]
# )
# self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
#
# def forward(self, x):
# fmap = []
#
# # 1d to 2d
# b, c, t = x.shape
# if t % self.period != 0:  # pad first
# n_pad = self.period - (t % self.period)
# x = F.pad(x, (0, n_pad), "reflect")
# t = t + n_pad
# x = x.view(b, c, t // self.period, self.period)
#
# for l in self.convs:
# x = l(x)
# x = F.leaky_relu(x, LRELU_SLOPE)
# fmap.append(x)
# x = self.conv_post(x)
# fmap.append(x)
# x = torch.flatten(x, 1, -1)
#
# return x, fmap
#
#
# class MultiPeriodDiscriminator(torch.nn.Module):
# def __init__(self):
# super(MultiPeriodDiscriminator, self).__init__()
# self.discriminators = nn.ModuleList(
# [
# DiscriminatorP(2),
# DiscriminatorP(3),
# DiscriminatorP(5),
# DiscriminatorP(7),
# DiscriminatorP(11),
# ]
# )
#
# def forward(self, y, y_hat):
# y_d_rs = []
# y_d_gs = []
# fmap_rs = []
# fmap_gs = []
# for i, d in enumerate(self.discriminators):
# y_d_r, fmap_r = d(y)
# y_d_g, fmap_g = d(y_hat)
# y_d_rs.append(y_d_r)
# fmap_rs.append(fmap_r)
# y_d_gs.append(y_d_g)
# fmap_gs.append(fmap_g)
#
# return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator module adapted from https://github.com/jik876/hifi-gan.
    Additionally, it allows incorporating conditional information with a learned embeddings table.

    Args:
        periods (tuple[int]): Tuple of periods for each discriminator.
        num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
            Defaults to None.
    """

    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=p, num_embeddings=num_embeddings) for p in periods]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        bandwidth_id: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        in_channels,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        (1, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=1024
            )
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x = x.unsqueeze(1)
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        num_embeddings: Optional[int] = None,
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
            num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
                Defaults to None.
        """

        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(window_length=w, num_embeddings=num_embeddings)
                for w in fft_sizes
            ]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        window_length: int,
        num_embeddings: Optional[int] = None,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            win_length=window_length,
            power=None,
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))
                ),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=channels
            )
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(
            nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1))
        )

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b f t c -> b c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        x = x.squeeze(1)
        x_bands = self.spectrogram(x)
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h

        return x, fmap


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False
    ):
        super(WavLMDiscriminator, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x
