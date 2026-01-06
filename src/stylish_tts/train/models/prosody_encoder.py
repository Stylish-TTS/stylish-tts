import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm
from .text_encoder import FFN, MultiHeadAttention, sequence_mask
from .ada_norm import AdaptiveLayerNorm


class ProsodyEncoder(nn.Module):
    def __init__(
        self,
        sty_dim,
        d_model,
        nlayers,
        dropout=0.1,
        n_heads=2,
        kernel_size=1,
        **kwargs,
    ):
        super().__init__()
        hidden_channels = d_model
        self.n_heads = n_heads
        self.n_layers = nlayers
        self.kernel_size = kernel_size

        self.drop = torch.nn.Dropout(dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        self.proj_layers = torch.nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=dropout
                )
            )
            self.norm_layers_1.append(
                AdaptiveLayerNorm(
                    style_dim=sty_dim,
                    channels=hidden_channels,
                )
            )
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(
                AdaptiveLayerNorm(
                    style_dim=sty_dim,
                    channels=hidden_channels,
                )
            )
            self.proj_layers.append(nn.Conv1d(hidden_channels, d_model, 1))

    def forward(self, x, style):
        # x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        # attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x_mask = 1
        attn_mask = None
        s = style
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i]((x + y).transpose(1, 2), s).transpose(1, 2)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i]((x + y).transpose(1, 2), s).transpose(1, 2)
            x = self.proj_layers[i](x)
        x = x * x_mask
        return x
