import math
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torchaudio.models import Conformer


class LearnedDownSample(torch.nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = torch.nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                torch.nn.Conv2d(
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
                torch.nn.Conv2d(
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


class DownSample(torch.nn.Module):
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


class ResBlk(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=torch.nn.LeakyReLU(0.2),
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
        self.conv1 = spectral_norm(torch.nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(torch.nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = torch.nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = torch.nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                torch.nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
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


class StyleEncoder(torch.nn.Module):
    def __init__(
        self, dim_in=48, style_dim=48, max_conv_dim=384, skip_downsamples=False
    ):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(torch.nn.Conv2d(1, dim_in, 3, 1, 1))]

        dim_out = 0
        repeat_num = 4
        for i in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            down = "half"
            if i == repeat_num - 1 and skip_downsamples:
                down = "none"
            blocks += [ResBlk(dim_in, dim_out, downsample=down)]
            dim_in = dim_out

        blocks += [torch.nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(torch.nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [torch.nn.AdaptiveAvgPool2d(1)]
        blocks += [torch.nn.LeakyReLU(0.2)]
        self.shared = torch.nn.Sequential(*blocks)

        self.unshared = torch.nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)

        return s


# class StyleEncoder(torch.nn.Module):
#     def __init__(self, *, mel_dim, style_dim, hidden_dim, num_heads, num_layers):
#         super().__init__()
#         self.mel_proj = torch.nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1)
#         self.conformer_pre = Conformer(
#             input_dim=hidden_dim,
#             num_heads=num_heads,
#             ffn_dim=hidden_dim * 2,
#             num_layers=1,
#             depthwise_conv_kernel_size=31,
#             use_group_norm=True,
#         )
#         self.conformer_body = Conformer(
#             input_dim=hidden_dim,
#             num_heads=num_heads,
#             ffn_dim=hidden_dim * 2,
#             num_layers=num_layers - 1,
#             depthwise_conv_kernel_size=15,
#             use_group_norm=True,
#         )
#         self.out = torch.nn.Linear(hidden_dim, style_dim)
#
#     def forward(self, mel, mel_lengths):
#         x = self.mel_proj(mel)
#         x = x.transpose(-1, -2)
#         x, output_lengths = self.conformer_pre(x, mel_lengths)
#         x, output_lengths = self.conformer_body(x, mel_lengths)
#         x = x.transpose(-1, -2)
#         x = self.out(x.mean(axis=-1))
#         return x
