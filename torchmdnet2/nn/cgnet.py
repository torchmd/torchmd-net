from torch import nn


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

    def __init__(self, model, baseline, train_baseline=False):
        super(CGnet, self).__init__()
        self.model = model
        self.baseline = baseline
        self.train_baseline = train_baseline

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

        energy = self.model(data.z, data.pos, data.batch)

        if not self.train_baseline and self.training:
          pass
        else:
          energy += self.baseline(data)
          
        force = -torch.autograd.grad(energy,
                                    data.pos,
                                     grad_outputs=torch.ones_like(energy),
                                    create_graph=True,
                                    retain_graph=True)[0]

        return energy, force