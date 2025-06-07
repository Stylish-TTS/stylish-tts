import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm
from .text_encoder import FFN, MultiHeadAttention, sequence_mask


# class DurationPredictor(nn.Module):
#     def __init__(self, style_dim, in_channels, filter_channels, kernel_size, dropout):
#         super().__init__()
#         self.in_channels = in_channels
#         self.filter_channels = filter_channels
#         self.dropout = dropout
#
#         self.drop = torch.nn.Dropout(dropout)
#         self.conv_1 = torch.nn.Conv1d(
#             in_channels, filter_channels, kernel_size, padding=kernel_size // 2
#         )
#         self.norm_1 = AdaLayerNorm(style_dim, filter_channels)
#         self.conv_2 = torch.nn.Conv1d(
#             filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
#         )
#         self.norm_2 = AdaLayerNorm(style_dim, filter_channels)
#         self.proj = torch.nn.Conv1d(filter_channels, 1, 1)
#
#     def forward(self, x, style, x_mask):
#         x = self.conv_1(x * x_mask)
#         x = torch.relu(x)
#         x = rearrange(x, "b c t -> b t c")
#         x = self.norm_1(x, style)
#         x = rearrange(x, "b t c -> b c t")
#         x = self.drop(x)
#         x = self.conv_2(x * x_mask)
#         x = torch.relu(x)
#         x = rearrange(x, "b c t -> b t c")
#         x = self.norm_2(x, style)
#         x = rearrange(x, "b t c -> b c t")
#         x = self.drop(x)
#         x = self.proj(x * x_mask)
#         return x * x_mask


class DurationPredictor(nn.Module):
    def __init__(
        self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1, has_duration_proj=True
    ):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.duration_proj = LinearNorm(d_hid + style_dim, max_dur)

    def forward(self, texts, style, text_lengths):
        d = self.text_encoder(texts, style, text_lengths)
        duration = self.duration_proj(d)
        return duration.squeeze(-1)


# class DurationEncoder(nn.Module):
#     def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
#         super().__init__()
#         self.lstms = nn.ModuleList()
#         for _ in range(nlayers):
#             self.lstms.append(
#                 nn.LSTM(
#                     d_model + sty_dim,
#                     d_model // 2,
#                     num_layers=1,
#                     batch_first=True,
#                     bidirectional=True,
#                     dropout=0,
#                 )
#             )
#             self.lstms.append(AdaLayerNorm(sty_dim, d_model))

#         self.dropout = dropout
#         self.d_model = d_model
#         self.sty_dim = sty_dim

#     def forward(self, x, style, text_lengths, m):
#         masks = m.to(text_lengths.device)

#         x = torch.cat([x, style], dim=1)
#         x = x.permute(2, 0, 1)
#         x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

#         x = x.transpose(0, 1)
#         input_lengths = text_lengths.cpu()
#         x = x.transpose(-1, -2)

#         for block in self.lstms:
#             if isinstance(block, AdaLayerNorm):
#                 x = block(x.transpose(-1, -2), style).transpose(-1, -2)
#                 x = torch.cat([x, style], dim=1)
#                 x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
#             else:
#                 x = x.transpose(-1, -2)
#                 x = nn.utils.rnn.pack_padded_sequence(
#                     x, input_lengths, batch_first=True, enforce_sorted=False
#                 )
#                 block.flatten_parameters()
#                 x, _ = block(x)
#                 x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#                 x = x.transpose(-1, -2)

#                 x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

#                 x_pad[:, :, : x.shape[-1]] = x
#                 x = x_pad.to(x.device)

#         return x.transpose(-1, -2)

#     def infer(self, x, style):
#         """
#         x: (batch, channels, tokens)
#         style: (batch, embedding)
#         """
#         # s = rearrange(style, "b e -> b e 1")
#         # s = s.expand(-1, -1, x.shape[2])  # batch embedding tokens
#         x = torch.cat([x, style], dim=1)

#         for block in self.lstms:
#             if isinstance(block, AdaLayerNorm):
#                 x = rearrange(x, "b c t -> b t c")
#                 x = block(x, style)
#                 x = rearrange(x, "b t c -> b c t")
#                 x = torch.cat([x, style], dim=1)
#             else:
#                 x = rearrange(x, "1 c t -> t c")
#                 block.flatten_parameters()
#                 x, _ = block(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#                 x = rearrange(x, "t c -> 1 c t")

#         return rearrange(x, "b c t -> b t c")


class DurationEncoder(nn.Module):
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
        hidden_channels = d_model + sty_dim
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
            self.norm_layers_1.append(AdaLayerNorm(sty_dim, hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(AdaLayerNorm(sty_dim, hidden_channels))
            self.proj_layers.append(nn.Conv1d(hidden_channels, d_model, 1))

    def forward(self, x, style, x_lengths):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = torch.cat([x, style], dim=1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )
            x = self.proj_layers[i](x)
            x = torch.cat([x, style], dim=1)
        x = x * x_mask
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

        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        gamma = h[:, :, : self.channels]
        beta = h[:, :, self.channels :]

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x
