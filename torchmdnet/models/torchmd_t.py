# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
)
from torchmdnet.utils import deprecated_class


@deprecated_class
class TorchMD_T(nn.Module):
    r"""The TorchMD Transformer architecture.
        This function optionally supports periodic boundary conditions with
        arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
        certain requirements:

        `a[1] = a[2] = b[2] = 0`
        `a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff`
        `a[0] >= 2*b[0]`
        `a[0] >= 2*c[0]`
        `b[1] >= 2*c[1]`

        These requirements correspond to a particular rotation of the system and
        reduced form of the vectors, as well as the requirement that the cutoff be
        no larger than half the box width.
    This model is considered deprecated and will be removed in a future release.
    Please refer to https://github.com/torchmd/torchmd-net/pull/240 for more details.

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
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
        box_vecs (Tensor, optional):
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            (default: :obj:`None`)
        check_errors (bool, optional): Whether to check for errors in the distance module.
            (default: :obj:`True`)

    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        check_errors=True,
        max_z=100,
        max_num_neighbors=32,
        dtype=torch.float,
        box_vecs=None,
    ):
        super(TorchMD_T, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

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
        self.max_z = max_z

        act_class = act_class_mapping[activation]
        attn_act_class = act_class_mapping[attn_activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels, dtype=dtype)

        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            loop=True,
            box=box_vecs,
            long_edge_index=True,
            check_errors=check_errors,
        )

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf, dtype=dtype
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
                dtype=dtype,
            )
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = MultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_act_class,
                cutoff_lower,
                cutoff_upper,
                dtype=dtype,
            )
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        x = self.embedding(z)

        edge_index, edge_weight, _ = self.distance(pos, batch, box)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        for attn in self.attention_layers:
            x = x + attn(x, edge_index, edge_weight, edge_attr, z.shape[0])
        x = self.out_norm(x)

        return x, None, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.

    :meta private:
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        dtype=torch.float,
    ):
        super(MultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.act = activation()
        self.attn_activation = attn_activation()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

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

    def forward(
        self, x: Tensor, edge_index: Tensor, r_ij: Tensor, f_ij: Tensor, n_atoms: int
    ) -> Tensor:
        head_shape = (-1, self.num_heads, self.head_dim)

        x = self.layernorm(x)
        q = self.q_proj(x).reshape(head_shape)
        k = self.k_proj(x).reshape(head_shape)
        v = self.v_proj(x).reshape(head_shape)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(head_shape)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(head_shape)
            if self.dv_proj is not None
            else None
        )
        msg = self.message(edge_index, q, k, v, dk, dv, r_ij)
        out = scatter(msg, edge_index[0], dim=0, dim_size=n_atoms)
        out = self.o_proj(out.reshape(-1, self.num_heads * self.head_dim))
        return out

    def message(
        self,
        edge_index: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
    ) -> Tensor:
        q_i = q.index_select(0, edge_index[0])
        k_j = k.index_select(0, edge_index[1])
        v_j = v.index_select(0, edge_index[1])
        # compute attention matrix
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        # weighted sum over values
        if dv is not None:
            v_j = v_j * dv
        v_j = v_j * attn.unsqueeze(2)
        return v_j
