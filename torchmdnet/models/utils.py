import ase
import math
import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class OutputNetwork(nn.Module):
    def __init__(self, hidden_channels, activation, readout='add', dipole=False,
                 mean=None, std=None, atomref=None, derivative=False, atom_filter=0,
                 max_z=100):
        super(OutputNetwork, self).__init__()

        if derivative:
            assert atom_filter == 0, 'Atom filter currently does not work if derivative is set to true'

        self.hidden_channels = hidden_channels
        self.activation = activation
        self.readout = readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.derivative = derivative
        self.atom_filter = atom_filter
        self.max_z = max_z

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = activation()
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(self.max_z, 1)
            self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, h, pos, batch):
        if not self.derivative:
            # drop atoms according to the filter
            atom_mask = z > self.atom_filter
            h = h[atom_mask]
            z = z[atom_mask]
            pos = pos[atom_mask]
            batch = batch[atom_mask]

        # output network
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.derivative:
            dy = -torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out),
                                      create_graph=True, retain_graph=True)[0]
            return out, dy
        return out


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr='add')
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]).item()**2

        if trainable:
            self.register_parameter('coeff', nn.Parameter(torch.scalar_tensor(coeff)))
            self.register_parameter('offset', nn.Parameter(offset))
        else:
            self.register_buffer('coeff', torch.scalar_tensor(coeff))
            self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.register_buffer('cutoff_lower', torch.scalar_tensor(cutoff_lower))

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)

        # initialize means and betas according to the default values in PhysNet (https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181)
        means = torch.linspace(torch.exp(torch.scalar_tensor(-cutoff_upper + cutoff_lower)), 1, num_rbf)
        betas = torch.tensor([(2 / num_rbf * (1 - torch.exp(torch.scalar_tensor(-cutoff_upper + cutoff_lower)))) ** -2] * num_rbf)

        if trainable:
            self.register_parameter('means', nn.Parameter(means))
            self.register_parameter('betas', nn.Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        return self.cutoff_fn(distances) * torch.exp(-self.betas * (torch.exp(-distances + self.cutoff_lower) - self.means) ** 2)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer('cutoff_lower', torch.scalar_tensor(cutoff_lower))
        self.register_buffer('cutoff_upper', torch.scalar_tensor(cutoff_upper))

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(math.pi * ( 2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)) + 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


rbf_class_mapping = {
    'gauss': GaussianSmearing,
    'expnorm': ExpNormalSmearing
}

act_class_mapping = {
    'ssp': ShiftedSoftplus,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'signmoid': nn.Sigmoid
}
