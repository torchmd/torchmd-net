import torch.nn as nn
import torch
import math

class ShiftedCosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff.

    .. math::
        sdf

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0, smooth_width=0.5):
        super(ShiftedCosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("smooth_width", torch.FloatTensor([smooth_width]))

    def forward(self, distances):
        """Compute cutoff function.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = torch.ones_like(distances)
        mask = distances > self.cutoff - self.smooth_width
        cutoffs[mask] = 0.5 + 0.5 * torch.cos(math.pi * (distances[mask] - self.cutoff + self.smooth_width) / self.smooth_width)
        cutoffs[distances > self.cutoff] = 0.

        return cutoffs.view(-1)
        