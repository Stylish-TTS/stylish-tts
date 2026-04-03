import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torchaudio.models import Conformer
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import Conv2d
from einops import rearrange
from .common import get_padding
from torch.nn import functional as F


class SpecDiscriminator(torch.nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm
        self.discriminators = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                    )
                ),
            ]
        )
        self.relu = torch.nn.LeakyReLU(0.1)

        self.out = torch.nn.ModuleList(
            [norm_f(torch.nn.Conv2d(32, 1, 3, 1, 1)) for _ in range(5)]
        )

    def forward(self, y):

        # fmap = []
        result = []
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = self.relu(y)
            out = self.out[i](y)
            out = torch.flatten(out, 1, -1)
            result.append(out)
            # fmap.append(y)

        # y = self.out(y)
        # fmap.append(y)

        # return [torch.flatten(y, 1, -1)], fmap
        return result, []


def run_discriminator_model(disc, target, pred):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []

    y_d_r, fmap_r = disc(target)
    y_d_g, fmap_g = disc(pred)
    y_d_rs = y_d_r
    # y_d_rs.append(y_d_r)
    fmap_rs.append(fmap_r)
    # y_d_gs.append(y_d_g)
    y_d_gs = y_d_g
    fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


#################################################
# Based on the model architecture in: https://arxiv.org/pdf/2508.15316


class ContextFreeBlock(torch.nn.Module):
    def __init__(
        self, dim_in, dim_out, *, kernel, groups=1, stride=1, dropout=0.0, bias=False
    ):
        super(ContextFreeBlock, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                dim_in,
                dim_out,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=groups,
                bias=bias,
            ),
            torch.nn.BatchNorm1d(num_features=dim_out),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ContextFreeDiscriminator(torch.nn.Module):

    def __init__(
        self,
    ):
        super(ContextFreeDiscriminator, self).__init__()
        dim = 64
        self.conv = torch.nn.ModuleList(
            [
                ContextFreeBlock(1, dim, kernel=11, stride=4, dropout=0.1),
                ContextFreeBlock(dim, dim * 2, kernel=11, stride=4, dropout=0.1),
                ContextFreeBlock(dim * 2, dim * 4, kernel=7, stride=2, dropout=0.1),
                ContextFreeBlock(dim * 4, dim * 4, kernel=5, stride=2, dropout=0.1),
            ]
        )
        self.attn = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(output_size=1),
            torch.nn.Conv1d(dim * 4, dim * 4, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )
        self.temporal = torch.nn.Sequential(
            ContextFreeBlock(
                dim * 4, dim * 4, kernel=7, groups=8, dropout=0.0, bias=True
            ),
            ContextFreeBlock(
                dim * 4, dim * 4, kernel=3, groups=8, dropout=0.0, bias=True
            ),
        )
        self.spectral = torch.nn.Sequential(
            ContextFreeBlock(
                dim * 4, dim * 12, kernel=1, groups=8, dropout=0.0, bias=True
            ),
            ContextFreeBlock(
                dim * 12, dim * 4, kernel=1, groups=8, dropout=0.0, bias=True
            ),
        )
        self.fusion = ContextFreeBlock(
            dim * 4 * 2, dim * 4, kernel=1, stride=1, dropout=0.1, bias=True
        )
        self.last = torch.nn.Sequential(
            torch.nn.Conv1d(dim * 2 * 2, dim * 2 * 4, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(dim * 2 * 4, 1, 1),
        )

    def forward(self, x):
        x = x.unfold(dimension=1, size=1024, step=512)
        time_steps = x.shape[1]
        x = rearrange(x, "b t w -> (b t) 1 w")
        for conv in self.conv:
            x = conv(x)
        attn = self.attn(x)
        x = x * attn
        temporal = self.temporal(x)
        spectral = self.spectral(x)
        x = torch.cat([temporal, spectral], dim=1)
        x = self.fusion(x)
        x = self.last(x)
        x = rearrange(x, "(b t) c f -> b (t c f)", t=time_steps)
        return [x], []


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = self.relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y.unsqueeze(1))
            y_d_rs.append(y_d_r)
            for item in fmap_r:
                fmap_rs.append(item)

        y_d_rs = torch.cat(y_d_rs, dim=1)
        return y_d_rs, fmap_rs
