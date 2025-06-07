import torch
from .conv_next import BasicConvNeXtBlock


class FineStyleEncoder(torch.nn.Module):
    def __init__(self, inter_dim, style_dim, layers):
        super().__init__()
        self.conv_in = torch.nn.Conv1d(inter_dim, style_dim, kernel_size=7, padding=3)
        self.blocks = torch.nn.ModuleList(
            [
                BasicConvNeXtBlock(
                    dim=style_dim,
                    intermediate_dim=style_dim * 4,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        strict = False
        return super().load_state_dict(state_dict, strict, assign)
