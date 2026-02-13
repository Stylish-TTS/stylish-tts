import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class PitchDiscriminator(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim_hidden,
        kernel,
    ):
        super(PitchDiscriminator, self).__init__()
        padding = kernel // 2
        self.discriminators = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        dim_in, dim_hidden, kernel_size=kernel, padding=padding
                    )
                ),
                weight_norm(
                    torch.nn.Conv1d(
                        dim_hidden, dim_hidden, kernel_size=kernel, padding=padding
                    )
                ),
                weight_norm(
                    torch.nn.Conv1d(
                        dim_hidden, dim_hidden, kernel_size=kernel, padding=padding
                    )
                ),
                weight_norm(
                    torch.nn.Conv1d(
                        dim_hidden, dim_hidden, kernel_size=kernel, padding=padding
                    )
                ),
                weight_norm(
                    torch.nn.Conv1d(
                        dim_hidden, dim_hidden, kernel_size=kernel, padding=padding
                    )
                ),
            ]
        )
        self.out = weight_norm(
            torch.nn.Conv1d(dim_hidden, 1, kernel_size=kernel, padding=padding)
        )

    def forward(self, y):
        # y = y.unsqueeze(1)
        fmap = []
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, 0.1)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap
