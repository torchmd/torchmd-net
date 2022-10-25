from abc import abstractmethod, ABCMeta
from torch import nn


class BasePrior(nn.Module, metaclass=ABCMeta):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.atomref.Atomref` prior.
    """

    def __init__(self, dataset=None):
        super().__init__()

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
