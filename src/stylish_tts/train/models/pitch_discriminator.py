import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class PitchDiscriminator(torch.nn.Module):
    def __init__(
        self,
    ):
        super(PitchDiscriminator, self).__init__()
        dim = 64
        self.discriminators = torch.nn.ModuleList(
            [
                weight_norm(torch.nn.Conv1d(3, dim, kernel_size=5, padding=1)),
                weight_norm(torch.nn.Conv1d(dim, dim, kernel_size=5, padding=1)),
                weight_norm(torch.nn.Conv1d(dim, dim, kernel_size=5, padding=1)),
                weight_norm(torch.nn.Conv1d(dim, dim, kernel_size=5, padding=1)),
                weight_norm(torch.nn.Conv1d(dim, dim, kernel_size=5, padding=1)),
            ]
        )
        self.out = weight_norm(torch.nn.Conv1d(dim, 1, kernel_size=5, padding=1))

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
