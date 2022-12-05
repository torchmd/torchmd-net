from torch import nn


class BasePrior(nn.Module):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.atomref.Atomref` prior.
    """

    def __init__(self, dataset=None):
        super().__init__()

    def get_init_args(self):
        r"""A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return {}

    def pre_reduce(self, x, z, pos, batch, extra_args):
        r"""Pre-reduce method of the prior model.

        Args:
            x (torch.Tensor): scalar atom-wise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.
            extra_args (dict): any addition fields provided by the dataset

        Returns:
            torch.Tensor: updated scalar atom-wise predictions
        """
        return x

    def post_reduce(self, y, z, pos, batch, extra_args):
        r"""Post-reduce method of the prior model.

        Args:
            y (torch.Tensor): scalar molecule-wise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.
            extra_args (dict): any addition fields provided by the dataset

        Returns:
            torch.Tensor: updated scalar molecular-wise predictions
        """
        return y
