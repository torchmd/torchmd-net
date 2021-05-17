import ase
from abc import abstractmethod, ABCMeta
from torch_scatter import scatter
from torchmdnet.models.utils import act_class_mapping

import torch
from torch import nn
from torch.autograd import grad


class BaseWrapper(nn.Module, metaclass=ABCMeta):
    r"""Base class for model wrappers.

    Children of this class should implement the `forward` method,
    which calls `self.model(z, pos, batch=batch)` at some point.
    Wrappers that are applied before the REDUCE operation should return
    the model's output, `z`, `pos` and `batch`. Wrappers that are applied
    after REDUCE should only return the model's output.
    """
    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    @abstractmethod
    def forward(self, z, pos, batch=None):
        return


class Derivative(BaseWrapper):
    def forward(self, z, pos, batch=None):
        pos.requires_grad_(True)
        out = self.model(z, pos, batch=batch)
        dy = -grad(out, pos, grad_outputs=torch.ones_like(out),
                   create_graph=True, retain_graph=True)[0]
        return out, dy


class Standardize(BaseWrapper):
    def __init__(self, model, mean, std):
        super(Standardize, self).__init__(model)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, z, pos, batch=None):
        out = self.model(z, pos, batch=batch)
        if self.std is not None:
            out = out * self.std
        if self.mean is not None:
            out = out + self.mean
        return out


class OutputNetwork(BaseWrapper):
    def __init__(self, model, hidden_channels, activation='silu',
                 reduce_op='add', dipole=False, prior_model=None):
        super(OutputNetwork, self).__init__(model)
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.reduce_op = reduce_op
        self.dipole = dipole
        self.prior_model = prior_model

        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer('atomic_mass', atomic_mass)

        act_class = act_class_mapping[activation]

        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        super(OutputNetwork, self).reset_parameters()
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        # run the potentially wrapped representation model
        x, z, pos, batch = self.model(z, pos, batch=batch)
        x = self.output_network(x)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            x = x * (pos - c[batch])
        elif self.prior_model is not None:
            # apply prior model
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)
        return out


class AtomFilter(BaseWrapper):
    def __init__(self, model, remove_threshold):
        super(AtomFilter, self).__init__(model)
        self.remove_threshold = remove_threshold

    def forward(self, z, pos, batch=None):
        x, z, pos, batch = self.model(z, pos, batch=batch)

        n_samples = len(batch.unique())

        # drop atoms according to the filter
        atom_mask = z > self.remove_threshold
        x = x[atom_mask]
        z = z[atom_mask]
        pos = pos[atom_mask]
        batch = batch[atom_mask]

        assert len(batch.unique()) == n_samples,\
            ('Some samples were completely filtered out by the atom filter. '
             f'Make sure that at least one atom per sample exists with Z > {self.remove_threshold}.')
        return x, z, pos, batch
