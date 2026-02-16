import math
import torch
from .text_encoder import TextEncoder
from .prosody_encoder import ProsodyEncoder
from stylish_tts.train.utils import length_to_mask
from .ada_norm import AdaptiveDecoderBlock


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        inter_dim,
        text_config,
        duration_config,
        pitch_energy_config,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(
            inter_dim=inter_dim,
            config=text_config,
        )
        dropout = pitch_energy_config.dropout
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=3,
            dropout=0.2,
        )

        d_hid = inter_dim
        self.F0 = torch.nn.ModuleList()
        self.F0.append(
            AdaptiveDecoderBlock(d_hid + style_dim, d_hid, style_dim, dropout_p=dropout)
        )
        self.F0.append(
            AdaptiveDecoderBlock(d_hid, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.F0.append(
            AdaptiveDecoderBlock(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.F0.append(
            AdaptiveDecoderBlock(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.N = torch.nn.ModuleList()
        self.N.append(
            AdaptiveDecoderBlock(d_hid + style_dim, d_hid, style_dim, dropout_p=dropout)
        )
        self.N.append(
            AdaptiveDecoderBlock(d_hid, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.N.append(
            AdaptiveDecoderBlock(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.N.append(
            AdaptiveDecoderBlock(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.F0_proj = torch.nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = torch.nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, text_lengths, alignment, style):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        mask = length_to_mask(text_lengths, text_encoding.shape[2]).to(
            text_lengths.device
        )
        prosody = self.prosody_encoder(text_encoding, style, text_lengths)  # , mask)

        x = prosody.transpose(1, 2) @ alignment
        x = x.transpose(-1, -2)

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)
