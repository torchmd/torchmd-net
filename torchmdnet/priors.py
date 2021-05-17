from abc import abstractmethod, ABCMeta
import torch
from torch import nn


class BasePrior(nn.Module, metaclass=ABCMeta):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which can be returned by the `prior_model`
    function of a dataset class. As an example, have a look at `torchmdnet.datasets.QM9`, which
    uses a `torchmdnet.priors.Atomref` prior.
    """

    def __init__(self):
        super(BasePrior, self).__init__()

    @abstractmethod
    def get_init_args(self):
        r"""A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return

    @abstractmethod
    def forward(self, x, z, pos, batch):
        r"""Forward method of the prior model.

        Args:
            x (torch.Tensor): scalar atomwise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.

        Returns:
            torch.Tensor: updated scalar atomwise predictions
        """
        return


class Atomref(BasePrior):
    def __init__(self, max_z, atomref=None):
        super(Atomref, self).__init__()
        if atomref is None:
            atomref = torch.zeros(max_z, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer('initial_atomref', atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(max_z=self.initial_atomref.size(0))

    def forward(self, x, z, pos, batch):
        return x + self.atomref(z)
