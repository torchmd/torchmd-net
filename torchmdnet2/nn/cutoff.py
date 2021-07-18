import torch.nn as nn
import torch



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
        cutoffs[distances > self.cutoff - self.smooth_width] = 0.5 + 0.5 * torch.cos(torch.pi * (distances - self.cutoff + self.smooth_width) / self.smooth_width)
        
        # Remove contributions beyond the cutoff radius
        cutoffs[distances > self.cutoff] = 0.

        return cutoffs