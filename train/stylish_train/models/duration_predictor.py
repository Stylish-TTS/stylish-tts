import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm


class DurationPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )

        self.lstm = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)

    def forward(self, texts, style, text_lengths, alignment, mask):
        d = self.text_encoder(texts, style, text_lengths, mask)

        # predict duration
        input_lengths = text_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False
        )

        mask = mask.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], mask.shape[-1], x.shape[-1]])

        x_pad[:, : x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(
            nn.functional.dropout(x, 0.5, training=self.training)
        )

        en = d.transpose(-1, -2) @ alignment

        return duration.squeeze(-1), en


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
                    dropout=0,
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
        input_lengths = text_lengths.cpu()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
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

    def infer(self, x, style):
        """
        x: (batch, channels, tokens)
        style: (batch, embedding)
        """
        s = rearrange(style, "b e -> b e 1")
        s = s.expand(-1, -1, x.shape[2])  # batch embedding tokens
        x = torch.cat([x, s], dim=1)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = rearrange(x, "b c t -> b t c")
                x = block(x, style)
                x = rearrange(x, "b t c -> b c t")
                x = torch.cat([x, s], dim=1)
            else:
                x = rearrange(x, "1 c t -> t c")
                block.flatten_parameters()
                x, _ = block(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = rearrange(x, "t c -> 1 c t")

        return rearrange(x, "b c t -> b t c")


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
