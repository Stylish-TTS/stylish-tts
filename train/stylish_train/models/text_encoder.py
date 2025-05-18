from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
from .common import LinearNorm, get_padding
from .conv_next import BasicConvNeXtBlock

from torchaudio.models import Conformer


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

        self.lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths, m, exteral_lstm: nn.LSTM = None):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        if exteral_lstm is None:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
        else:
            exteral_lstm.flatten_parameters()
            x, _ = exteral_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def infer(self, x, external_lstm: nn.LSTM = None):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]

        for c in self.cnn:
            x = c(x)

        x = rearrange(x, "1 c t -> t c")
        if external_lstm is None:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
        else:
            external_lstm.flatten_parameters()
            x, _ = external_lstm(x)
        x = rearrange(x, "t c -> 1 c t")
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
