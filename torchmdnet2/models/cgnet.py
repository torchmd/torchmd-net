from .schnet import SchNet
from torch import nn
from torch.autograd import grad
import torch

class CGnet(nn.Module):
    """CGnet neural network class

    Parameters
    ----------
    feature : nn.Module() instance
        feature layer to transform cartesian coordinates into roto-
        translationally invariant features.
    baseline : list of nn.Module() instances (default=None)
        list of prior layers that provide energy contributions external to
        the hidden architecture of the CGnet.
    train_baseline : bool
        use the baseline model within the training

    """

    def __init__(self, model, baseline):
        super(CGnet, self).__init__()
        self.model = model
        self.baseline = baseline

    def forward(self, data):
        """Forward pass through the network ending with autograd layer.

        Parameters
        ----------
        data : torch_geometric.data

        Returns
        -------
        energy : torch.Tensor
            scalar potential energy of size [n_frames, 1]. If priors are
            supplied to the CGnet, then this energy is the sum of network
            and prior energies.
        force  : torch.Tensor
            vector forces of size [n_atoms, 3].


        """
        data.pos.requires_grad_()

        if isinstance(self.model, SchNet):
            energy = self.model(data.z, data.pos, data.batch)
        else:
            energy = self.model(data)

        if isinstance(energy, tuple):
            energy, forces = energy
            baseline_energy = self.baseline(data.pos, data.idx.shape[0])
            forces += -grad(baseline_energy, data.pos,
                                     grad_outputs=torch.ones_like(baseline_energy),
                                    create_graph=True,
                                    retain_graph=True)[0]
        if isinstance(energy, dict):
            energy, forces = energy['energy'], energy['forces']
            baseline_energy = self.baseline(data.pos, data.idx.shape[0])
            forces += -grad(baseline_energy, data.pos,
                                     grad_outputs=torch.ones_like(baseline_energy),
                                    create_graph=True,
                                    retain_graph=True)[0]
        else:
            energy += self.baseline(data.pos, data.idx.shape[0])
            forces = -grad(energy,data.pos,
                                    grad_outputs=torch.ones_like(energy),
                                    create_graph=True,
                                    retain_graph=True)[0]


        return energy, forces