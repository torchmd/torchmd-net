from torchmdnet.priors.base import BasePrior
import torch
from torch import nn
from pytorch_lightning.utilities import rank_zero_warn


class Atomref(BasePrior):
    r"""Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, dataset=None):
        super().__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = dataset.get_atomref()
            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(max_z=self.initial_atomref.size(0))

    def pre_reduce(self, x, z, pos, batch, extra_args):
        return x + self.atomref(z)
