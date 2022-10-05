from abc import abstractmethod, ABCMeta
import torch
from torch import nn
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models.utils import Distance

__all__ = ["Atomref", "ZBL"]


class BasePrior(nn.Module, metaclass=ABCMeta):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.Atomref` prior.
    """

    def __init__(self, dataset=None, atomwise=True):
        super(BasePrior, self).__init__()
        self.atomwise = atomwise

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
            torch.Tensor: If this is an atom-wise prior (self.atomwise is True), the return value
            contains updated scalar atomwise predictions.  Otherwise, it is a single scalar that
            is added to the result after summing over atoms.
        """
        return


class Atomref(BasePrior):
    r"""Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, dataset=None):
        super(Atomref, self).__init__()
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

    def forward(self, x, z, pos, batch):
        return x + self.atomref(z)


class ZBL(BasePrior):
    """This class implements the Ziegler-Biersack-Littmark (ZBL) potential for screened nuclear repulsion.
    Is is described in https://doi.org/10.1007/978-3-642-68779-2_5 (equations 9 and 10 on page 147).  It
    is an empirical potential that does a good job of describing the repulsion between atoms at very short
    distances.

    To use this prior, the Dataset must provide the following attributes.

    atomic_number: 1D tensor of length max_z.  atomic_number[z] is the atomic number of atoms with atom type z.
    distance_scale: multiply by this factor to convert coordinates stored in the dataset to meters
    energy_scale: multiply by this factor to convert energies stored in the dataset to Joules
    """
    def __init__(self, dataset=None):
        super(ZBL, self).__init__(atomwise=False)
        self.register_buffer("atomic_number", dataset.atomic_number)
        self.distance = Distance(0, 10.0, max_num_neighbors=100)
        self.distance_scale = dataset.distance_scale*1.88973e10  # convert to Bohr units
        self.energy_scale = dataset.energy_scale  # convert to Joules

    def get_init_args(self):
        return {}

    def reset_parameters(self):
        pass

    def forward(self, x, z, pos, batch):
        edge_index, distance, _ = self.distance(pos*self.distance_scale, batch)
        atomic_number = self.atomic_number[z[edge_index]]
        a = 0.8854/(atomic_number[0]**0.23 + atomic_number[0]**0.23)
        d = distance/a
        f = 0.1818*torch.exp(-3.2*d) + 0.5099*torch.exp(-0.9423*d) + 0.2802*torch.exp(-0.4029*d) + 0.02817*torch.exp(-0.2016*d)
        return (2.30707755e-19/self.energy_scale)*torch.sum(f/distance, dim=-1)
