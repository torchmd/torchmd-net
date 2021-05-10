import torch
from torch import nn
from torch_geometric.nn import radius_graph, MessagePassing

from torchmdnet.models.utils import (OutputNetwork, NeighborEmbedding, CosineCutoff,
                                     rbf_class_mapping, act_class_mapping)


class TorchMD_T(nn.Module):
    r"""The TorchMD Transformer architecture.
    Code adapted from https://github.com/rusty1s/pytorch_geometric/blob/d7d8e5e2edada182d820bbb1eec5f016f50db1e0/torch_geometric/nn/models/schnet.py#L38

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
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
        atom_filter (int, optional): Only sum over atoms with Z > atom_filter.
            (default: :obj:`-1`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
    """

    def __init__(self, hidden_channels=128, num_layers=6, num_rbf=50, rbf_type='expnorm',
                 trainable_rbf=True, activation='silu', attn_activation='silu', neighbor_embedding=True,
                 num_heads=8, distance_influence='both', cutoff_lower=0.0, cutoff_upper=5.0,
                 readout='add', dipole=False, mean=None, std=None, atomref=None, derivative=False,
                 atom_filter=-1, max_z=100):
        super(TorchMD_T, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        assert distance_influence in ['keys', 'values', 'both']
        assert rbf_type in rbf_class_mapping, (f'Unknown RBF type "{rbf_type}". '
                                               f'Choose from {", ".join(rbf_class_mapping.keys())}.')
        assert activation in act_class_mapping, (f'Unknown activation function "{activation}". '
                                                 f'Choose from {", ".join(act_class_mapping.keys())}.')

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.readout = 'add' if dipole else readout
        self.dipole = dipole
        self.derivative = derivative
        self.atom_filter = atom_filter
        self.max_z = max_z

        act_class = act_class_mapping[activation]
        attn_act_class = act_class_mapping[attn_activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
        ) if neighbor_embedding else None

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = MultiHeadAttention(hidden_channels, num_rbf, distance_influence,
                                       act_class, attn_act_class, cutoff_lower, cutoff_upper)
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.output_network = OutputNetwork(
            hidden_channels, act_class, self.readout, self.dipole, mean, std,
            atomref, self.derivative, self.atom_filter, self.max_z
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.output_network.reset_parameters()

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff_upper, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding:
            h = self.neighbor_embedding(z, h, edge_index, edge_weight, edge_attr)

        for attn in self.attention_layers:
            h = h + attn(h, edge_index, edge_weight, edge_attr)

        h = self.out_norm(h)
        
        return self.output_network(z, h, pos, batch)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_rbf={self.num_rbf}, '
                f'rbf_type={self.rbf_type}, '
                f'trainable_rbf={self.trainable_rbf}, '
                f'activation={self.activation}, '
                f'attn_activation={self.attn_activation}, '
                f'neighbor_embedding={self.neighbor_embedding}, '
                f'num_heads={self.num_heads}, '
                f'distance_influence={self.distance_influence}, '
                f'cutoff_lower={self.cutoff_lower}, '
                f'cutoff_upper={self.cutoff_upper}, '
                f'derivative={self.derivative}, '
                f'atom_filter={self.atom_filter})')


class MultiHeadAttention(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, distance_influence,
                 activation, attn_activation, cutoff_lower, cutoff_upper):
        super(MultiHeadAttention, self).__init__(aggr='add')
        self.distance_influence = distance_influence

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.activation = activation()
        self.attn_activation = attn_activation()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dk_proj = None
        if distance_influence in ['keys', 'both']:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.dv_proj = None
        if distance_influence in ['values', 'both']:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x_norm = self.layernorm(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        dk = self.activation(self.dk_proj(edge_attr)) if self.dk_proj else 1.0
        dv = self.activation(self.dv_proj(edge_attr)) if self.dv_proj else 1.0
        
        out = self.propagate(edge_index, q=q, k=k, v=v, dk=dk, dv=dv, r_ij=edge_weight)
        out = self.o_proj(out)
        return x + out

    def message(self, q_i, k_j, v_j, dk, dv, r_ij):
        attn = (q_i * k_j * dk).sum(dim=1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij)
        v_j = v_j * dv
        v_j = v_j * attn.unsqueeze(1)
        return v_j
