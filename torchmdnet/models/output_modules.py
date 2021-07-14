from abc import abstractmethod, ABCMeta
import ase
from torchmdnet.models.utils import act_class_mapping
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
import torch
from torch import nn
from torch.autograd import grad


__all__ = ['Scalar', 'Dipole']


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(self, is_equivariant, hidden_channels, activation='silu'):
        super(Scalar, self).__init__(allow_prior_model=True)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch):
        return self.output_network(x)


class Dipole(OutputModel):
    def __init__(self, is_equivariant, hidden_channels, activation='silu'):
        super(Dipole, self).__init__(allow_prior_model=False)
        self.is_equivariant = is_equivariant
        act_class = act_class_mapping[activation]

        if is_equivariant:
            self.output_network = nn.ModuleList([
                GatedEquivariantBlock(hidden_channels, hidden_channels // 2,
                                      activation=activation, scalar_activation=True),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ])
        else:
            self.output_network = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                act_class(),
                nn.Linear(hidden_channels // 2, 1)
            )

        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer('atomic_mass', atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_equivariant:
            self.output_network[0].reset_parameters()
            self.output_network[1].reset_parameters()
        else:
            nn.init.xavier_uniform_(self.output_network[0].weight)
            self.output_network[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.output_network[2].weight)
            self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch):
        if self.is_equivariant:
            for layer in self.output_network:
                x, v = layer(x, v)
        else:
            x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])

        if self.is_equivariant:
            x = x + v.squeeze()
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class TorchMD_Net(nn.Module):
    def __init__(self, representation_model, output_model, prior_model=None,
                 reduce_op='add', mean=None, std=None, derivative=False):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        if output_model.allow_prior_model:
            self.prior_model = prior_model
        else:
            self.prior_model = None
            rank_zero_warn(('Prior model was given but the output model does '
                            'not allow prior models. Dropping the prior model.'))

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer('mean', mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer('std', std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        representation = self.representation_model(z, pos, batch=batch)

        if len(representation) == 5:
            x, v, z, pos, batch = representation
        else:
            v = None
            x, z, pos, batch = representation

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # standardize if no prior model is given and the output model allows priors
        if self.prior_model is None and self.output_model.allow_prior_model:
            if self.std is not None:
                out = out * self.std
            if self.mean is not None:
                out = out + self.mean

        out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.derivative:
            dy = -grad(out, pos, grad_outputs=torch.ones_like(out),
                    create_graph=True, retain_graph=True)[0]
            return out, dy
        return out


class GatedEquivariantBlock(nn.Module):
    def __init__(self, hidden_channels, out_channels, intermediate_channels=None,
                 activation='silu', scalar_activation=False):
        super(GatedEquivariantBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.scalar_activation = scalar_activation

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2)
        )

        if scalar_activation:
            self.act = act_class()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)
        
        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.scalar_activation:
            x = self.act(x)
        return x, v
