import torch
from .conv_next import BasicConvNeXtBlock


class FineStyleEncoder(torch.nn.Module):
    # TODO: Remvoe hard-coded values
    def __init__(self, inter_dim, style_dim, config):
        super().__init__()
        self.conv_in = torch.nn.Conv1d(inter_dim, style_dim, kernel_size=7, padding=3)
        self.blocks = torch.nn.ModuleList(
            [
                BasicConvNeXtBlock(
                    dim=style_dim,
                    intermediate_dim=style_dim * 4,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x, lengths):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return x
