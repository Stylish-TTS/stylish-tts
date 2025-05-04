import torch
from torch.nn import functional as F
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    xLSTMBlockStack,
)
from .conv_next import ConvNeXtBlock


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(self, style_dim, d_hid, dropout=0.1):
        super().__init__()

        self.cfg_pred = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            context_length=4096,
            num_blocks=8,
            embedding_dim=d_hid + style_dim,
        )
        self.shared = xLSTMBlockStack(self.cfg_pred)
        self.prepare_projection = torch.nn.Linear(d_hid + style_dim, d_hid)

        self.F0 = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=d_hid,
                    dim_out=d_hid,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
                ConvNeXtBlock(
                    dim_in=d_hid,
                    dim_out=d_hid // 2,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
                ConvNeXtBlock(
                    dim_in=d_hid // 2,
                    dim_out=d_hid // 2,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
            ]
        )

        self.N = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=d_hid,
                    dim_out=d_hid,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
                ConvNeXtBlock(
                    dim_in=d_hid,
                    dim_out=d_hid // 2,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
                ConvNeXtBlock(
                    dim_in=d_hid // 2,
                    dim_out=d_hid // 2,
                    intermediate_dim=d_hid * 2,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                    dropout=dropout,
                ),
            ]
        )

        self.F0_proj = torch.nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = torch.nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, prosody, style):
        x = self.shared(prosody.transpose(-1, -2))
        x = self.prepare_projection(x)
        F0 = x.transpose(-1, -2)
        F0 = F.interpolate(F0, scale_factor=2, mode="nearest")
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        N = F.interpolate(N, scale_factor=2, mode="nearest")
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)
