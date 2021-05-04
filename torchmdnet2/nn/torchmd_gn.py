import ase
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
import numpy as np

from torch_scatter import scatter
from torch_geometric.nn import radius_graph, MessagePassing


class TorchMD_GN(torch.nn.Module):
    r"""The TorchMD Graph Network architecture.
    Code adapted from https://github.com/rusty1s/pytorch_geometric/blob/d7d8e5e2edada182d820bbb1eec5f016f50db1e0/torch_geometric/nn/models/schnet.py#L38

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        embedding_size (int, optional): Size of the embedding dictionary.
            (default :obj:`100`)
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
        derivative (bool, optional): If True, computes the derivative of the prediction
            w.r.t the input coordinates. (default: :obj:`False`)
        cfconv_aggr (str, optional): The aggregation method for CFConv filter
            outputs. (default: :obj:`mean`)
    """

    def __init__(self, embedding_size=100, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_rbf=50, rbf_type='expnorm',
                 trainable_rbf=True, activation='silu', neighbor_embedding=True,
                 cutoff_lower=0.0, cutoff_upper=5.0, readout='add', dipole=False, mlp_out=None,
                 mean=None, std=None, atomref=None, derivative=False,
                 cfconv_aggr='mean'):
        super(TorchMD_GN, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        assert cfconv_aggr in ['add', 'max', 'mean', None]
        assert rbf_type in rbf_class_mapping, f'Unknown RBF type "{rbf_type}". Choose from {", ".join(rbf_class_mapping.keys())}.'
        assert activation in act_class_mapping, f'Unknown activation function "{activation}". Choose from {", ".join(act_class_mapping.keys())}.'
        self.embedding_size = embedding_size
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.derivative = derivative
        self.cfconv_aggr = cfconv_aggr

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        act_class = act_class_mapping[activation]

        self.embedding = Embedding(self.embedding_size, hidden_channels)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff_lower, cutoff_upper, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff_lower, cutoff_upper) if neighbor_embedding else None

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_rbf, num_filters,
                                     act_class, cutoff_lower, cutoff_upper,
                                     cfconv_aggr=self.cfconv_aggr)
            self.interactions.append(block)
        if mlp_out is None:
            self.mlp = Sequential(
                Linear(hidden_channels, hidden_channels // 2),
                act_class(),
                Linear(hidden_channels // 2, 1)
            )
        else:
            self.mlp = mlp_out

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(self.embedding_size, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

        self.mlp.apply(self.init_weights)

        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_()

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff_upper, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding:
            h = self.neighbor_embedding(z, h, edge_index, edge_weight, edge_attr)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.mlp(h)

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

        if self.scale is not None:
            out = self.scale * out

        if self.derivative:
            dy = -torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]
            return out, dy

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_rbf={self.num_rbf}, '
                f'rbf_type={self.rbf_type}, '
                f'activation={self.activation}, '
                f'cutoff_lower={self.cutoff_lower}, '
                f'cutoff_upper={self.cutoff_upper}, '
                f'derivative={self.derivative})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_rbf, num_filters, activation,
                      cutoff_lower, cutoff_upper, cfconv_aggr='mean'):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_rbf, num_filters),
            activation(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff_lower, cutoff_upper,
                           aggr=cfconv_aggr)
        self.act = activation()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff_lower,
                 cutoff_upper, aggr='mean'):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper):
        super(NeighborEmbedding, self).__init__(aggr='add')
        self.embedding = Embedding(100, hidden_channels)
        self.distance_proj = Linear(num_rbf, hidden_channels)
        self.combine = Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        torch.nn.init.xavier_uniform_(self.distance_proj.weight)
        torch.nn.init.xavier_uniform_(self.combine.weight)
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


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]).item()**2

        if trainable:
            self.register_parameter('coeff', torch.nn.Parameter(torch.scalar_tensor(coeff)))
            self.register_parameter('offset', torch.nn.Parameter(offset))
        else:
            self.register_buffer('coeff', torch.scalar_tensor(coeff))
            self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.register_buffer('cutoff_lower', torch.scalar_tensor(cutoff_lower))

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)

        # initialize means and betas according to the default values in PhysNet (https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181)
        means = torch.linspace(torch.exp(torch.scalar_tensor(-cutoff_upper + cutoff_lower)), 1, num_rbf)
        betas = torch.tensor([(2 / num_rbf * (1 - torch.exp(torch.scalar_tensor(-cutoff_upper + cutoff_lower)))) ** -2] * num_rbf)

        if trainable:
            self.register_parameter('means', torch.nn.Parameter(means))
            self.register_parameter('betas', torch.nn.Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        return self.cutoff_fn(distances) * torch.exp(-self.betas * (torch.exp(-distances + self.cutoff_lower) - self.means) ** 2)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff_lower", torch.scalar_tensor(cutoff_lower))
        self.register_buffer("cutoff_upper", torch.scalar_tensor(cutoff_upper))

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(np.pi * ( 2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)) + 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class GraphNormMSE(torch.nn.Module):
    def __init__(self):
        super(GraphNormMSE, self).__init__()

    def forward(self, pred_prop, prop, batch):
        """Calculation of the MSE loss per graph.

        Args:
            pred_prop (torch.tensor): property predicted by the
                model, with shape (n_examples, **dims)
            prop (torch.tensor): labeled property that the model
                is attempting to match, with shape (n_examples, **dims)
            batch (torch.tensor): batch indices, as supplied by
                pytorch geometric dataloader, with shape (n_examples)
        Returns:
            loss (torch.tensor): MSE loss per graph, where each example
                has been normalized by the size of the graph from which
                it originated. Shape (1); scalar.
        """
        node_idx, node_sizes = torch.unique(batch, return_counts=True)
        example_sizes = node_sizes[batch]
        prop_diff = prop - pred_prop
        prop_diff = prop_diff * example_sizes[:, None]
        loss = (prop_diff**2).mean()
        return loss


rbf_class_mapping = {
    'gauss': GaussianSmearing,
    'expnorm': ExpNormalSmearing
}

act_class_mapping = {
    'ssp': ShiftedSoftplus,
    'silu': torch.nn.SiLU,
    'tanh': torch.nn.Tanh,
    'signmoid': torch.nn.Sigmoid
}
