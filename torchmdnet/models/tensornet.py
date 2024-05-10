# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    MLP,
    nvtx_annotate,
    nvtx_range,
)

__all__ = ["TensorNet"]
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True


@nvtx_annotate("vector_to_skewtensor")
def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.shape[:-1]
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[..., 2],
            vector[..., 1],
            vector[..., 2],
            zero,
            -vector[..., 0],
            -vector[..., 1],
            vector[..., 0],
            zero,
        ),
        dim=-1,
    )
    tensor = tensor.view(*batch_size, 3, 3)
    return tensor.squeeze(0)


@nvtx_annotate("vector_to_symtensor")
def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    S = 0.5 * (tensor + tensor.transpose(-2, -1))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)
    S.diagonal(offset=0, dim1=-1, dim2=-2).sub_(I.unsqueeze(-1))
    return S


@nvtx_annotate("decompose_tensor")
def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = tensor - A
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)
    S.diagonal(offset=0, dim1=-1, dim2=-2).sub_(I.unsqueeze(-1))
    return I, A, S


@nvtx_annotate("tensor_norm")
def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor**2).sum((-2, -1))


class TensorNet(nn.Module):
    r"""TensorNet's architecture. From
    TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials; G. Simeon and G. de Fabritiis.
    NeurIPS 2023.

    This function optionally supports periodic boundary conditions with arbitrary triclinic boxes.
    For a given cutoff, :math:`r_c`, the box vectors :math:`\vec{a},\vec{b},\vec{c}` must satisfy certain requirements:

    .. math::

      \begin{align*}
      a_y = a_z = b_z &= 0 \\
      a_x, b_y, c_z &\geq 2 r_c \\
      a_x &\geq 2  b_x \\
      a_x &\geq 2  c_x \\
      b_y &\geq 2  c_y
      \end{align*}

    These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of interaction layers.
            (default: :obj:`2`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`32`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`False`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`4.5`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`128`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            (default: :obj:`64`)
        equivariance_invariance_group (string, optional): Group under whose action on input
            positions internal tensor features will be equivariant and scalar predictions
            will be invariant. O(3) or SO(3).
            (default :obj:`"O(3)"`)
        box_vecs (Tensor, optional):
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            (default: :obj:`None`)
        static_shapes (bool, optional): Whether to enforce static shapes.
            Makes the model CUDA-graph compatible if check_errors is set to False.
            (default: :obj:`True`)
        check_errors (bool, optional): Whether to check for errors in the distance module.
            (default: :obj:`True`)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=2,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_lower=0,
        cutoff_upper=4.5,
        max_num_neighbors=64,
        max_z=128,
        equivariance_invariance_group="O(3)",
        static_shapes=True,
        check_errors=True,
        dtype=torch.float32,
        box_vecs=None,
    ):
        super(TensorNet, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], (
            f'Unknown group "{equivariance_invariance_group}". '
            f"Choose O(3) or SO(3)."
        )
        self.hidden_channels = hidden_channels
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        act_class = act_class_mapping[activation]
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.tensor_embedding = TensorEmbedding(
            hidden_channels,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            max_z,
            dtype,
        )

        self.layers = nn.ModuleList()
        if num_layers != 0:
            for _ in range(num_layers):
                self.layers.append(
                    Interaction(
                        num_rbf,
                        hidden_channels,
                        act_class,
                        cutoff_lower,
                        cutoff_upper,
                        equivariance_invariance_group,
                        dtype,
                    )
                )
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)
        self.out_norm = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
        self.act = act_class()
        # Resize to fit set to false ensures Distance returns a statically-shaped tensor of size max_num_pairs=pos.size*max_num_neigbors
        # negative max_num_pairs argument means "per particle"
        # long_edge_index set to False saves memory and spares some kernel launches by keeping neighbor indices as int32.
        self.static_shapes = static_shapes
        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            return_vecs=True,
            loop=True,
            check_errors=check_errors,
            resize_to_fit=not self.static_shapes,
            box=box_vecs,
            long_edge_index=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.linear.reset_parameters()
        self.out_norm.reset_parameters()

    @nvtx_annotate("make_static")
    def _make_static(
        self, num_nodes: int, edge_index: Tensor, edge_weight: Tensor, edge_vec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Distance module returns -1 for non-existing edges, to avoid having to resize the tensors when we want to ensure static shapes (for CUDA graphs) we make all non-existing edges pertain to a ghost atom
        if self.static_shapes:
            mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
            # I trick the model into thinking that the masked edges pertain to the extra atom
            # WARNING: This can hurt performance if max_num_pairs >> actual_num_pairs
            edge_index = edge_index.masked_fill(mask, num_nodes)
            edge_weight = edge_weight.masked_fill(mask[0], 0)
            edge_vec = edge_vec.masked_fill(
                mask[0].unsqueeze(-1).expand_as(edge_vec), 0
            )
        return edge_index, edge_weight, edge_vec

    @nvtx_annotate("compute_neighbors")
    def _compute_neighbors(
        self, pos: Tensor, batch: Tensor, box: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)
        # This assert convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        edge_index, edge_weight, edge_vec = self._make_static(
            pos.shape[0], edge_index, edge_weight, edge_vec
        )
        return edge_index, edge_weight, edge_vec

    @nvtx_annotate("output")
    def output(self, X: Tensor) -> Tensor:
        I, A, S = decompose_tensor(X)  # shape: (n_atoms, hidden_channels, 3, 3)
        x = torch.cat(
            (3 * I**2, tensor_norm(A), tensor_norm(S)), dim=-1
        )  # shape: (n_atoms, 3*hidden_channels)
        x = self.out_norm(x)  # shape: (n_atoms, 3*hidden_channels)
        x = self.act(self.linear((x)))  # shape: (n_atoms, hidden_channels)
        return x

    @nvtx_annotate("TensorNet")
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        if self.static_shapes:
            z = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)
        # Total charge q is a molecule-wise property. We transform it into an atom-wise property, with all atoms belonging to the same molecule being assigned the same charge q
        if q is None:
            q = torch.zeros_like(z, device=z.device, dtype=z.dtype)
        else:
            q = q[batch]
        edge_index, edge_weight, edge_vec = self._compute_neighbors(pos, batch, box)
        edge_attr = self.distance_expansion(edge_weight)  # shape: (n_edges, num_rbf)
        X = self.tensor_embedding(
            z, edge_index, edge_weight, edge_vec, edge_attr
        )  # shape: (n_atoms, hidden_channels, 3, 3)
        for layer in self.layers:
            X = layer(
                X, edge_index, edge_weight, edge_attr, q
            )  # shape: (n_atoms, hidden_channels, 3, 3)
        x = self.output(X)  # shape: (n_atoms, hidden_channels)
        # Remove the extra atom
        if self.static_shapes:
            x = x[:-1]
            z = z[:-1]
        return x, None, z, pos, batch


class TensorLinear(nn.Module):

    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super(TensorLinear, self).__init__()
        self.linearI = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype)
        self.linearA = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype)
        self.linearS = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype)

    def reset_parameters(self):
        self.linearI.reset_parameters()
        self.linearA.reset_parameters()
        self.linearS.reset_parameters()

    @nvtx_annotate("TensorLinear")
    def forward(self, X: Tensor, factor: Optional[Tensor] = None) -> Tensor:
        if factor is None:
            factor = (
                torch.ones(1, device=X.device, dtype=X.dtype)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).expand(-1, -1, 3)
        I, A, S = decompose_tensor(X)
        I = self.linearI(I) * factor[..., 0]
        A = (
            self.linearA(A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * factor[..., 1, None, None]
        )
        S = (
            self.linearS(S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * factor[..., 2, None, None]
        )
        dX = A + S
        dX.diagonal(dim1=-2, dim2=-1).add_(I.unsqueeze(-1))
        return dX


class TensorEmbedding(nn.Module):
    """Tensor embedding layer.

    :meta private:
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        activation,
        cutoff_lower,
        cutoff_upper,
        max_z=128,
        dtype=torch.float32,
    ):
        super(TensorEmbedding, self).__init__()

        self.hidden_channels = hidden_channels
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.emb2 = nn.Linear(2 * hidden_channels, hidden_channels, dtype=dtype)
        self.act = activation()
        self.linear_tensor = TensorLinear(hidden_channels, hidden_channels)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        self.linear_tensor.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    @nvtx_annotate("normalize_edges")
    def _normalize_edges(
        self, edge_index: Tensor, edge_weight: Tensor, edge_vec: Tensor
    ) -> Tensor:
        mask = edge_index[0] == edge_index[1]
        # Normalizing edge vectors by their length can result in NaNs, breaking Autograd.
        # I avoid dividing by zero by setting the weight of self edges and self loops to 1
        edge_vec = edge_vec / edge_weight.masked_fill(mask, 1).unsqueeze(1)
        return edge_vec

    @nvtx_annotate("compute_edge_atomic_features")
    def _compute_edge_atomic_features(self, z: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(z)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
                -1, self.hidden_channels * 2
            )
        )
        return Zij

    @nvtx_annotate("compute_edge_tensor_features")
    def _compute_node_tensor_features(
        self,
        z: Tensor,
        edge_index,
        edge_weight: Tensor,
        edge_vec: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        edge_vec_norm = self._normalize_edges(
            edge_index, edge_weight, edge_vec
        )  # shape: (n_edges, 3)
        Zij = self.cutoff(edge_weight)[:, None] * self._compute_edge_atomic_features(
            z, edge_index
        )  # shape: (n_edges, hidden_channels)

        A = (
            self.distance_proj2(edge_attr)[
                ..., None
            ]  # shape: (n_edges, hidden_channels, 1)
            * Zij[..., None]  # shape: (n_edges, hidden_channels, 1)
            * edge_vec_norm[:, None, :]  # shape: (n_edges, 1, 3)
        )  # shape: (n_edges, hidden_channels, 3)
        A = self._aggregate_edge_features(
            z.shape[0], A, edge_index[0]
        )  # shape: (n_atoms, hidden_channels, 3)
        A = vector_to_skewtensor(A)  # shape: (n_atoms, hidden_channels, 3, 3)

        S = (
            self.distance_proj3(edge_attr)[..., None, None]
            * Zij[..., None, None]
            * vector_to_symtensor(edge_vec_norm)[..., None, :, :]
        )  # shape: (n_edges, hidden_channels, 3, 3)
        S = self._aggregate_edge_features(
            z.shape[0], S, edge_index[0]
        )  # shape: (n_atoms, hidden_channels, 3, 3)
        I = self.distance_proj1(edge_attr) * Zij
        I = self._aggregate_edge_features(z.shape[0], I, edge_index[0])
        features = A + S
        features.diagonal(dim1=-2, dim2=-1).add_(I.unsqueeze(-1))
        return features

    @nvtx_annotate("aggregate_edge_features")
    def _aggregate_edge_features(
        self, num_atoms: int, T: Tensor, source_indices: Tensor
    ) -> Tensor:
        targetI = torch.zeros(num_atoms, *T.shape[1:], device=T.device, dtype=T.dtype)
        I = targetI.index_add(dim=0, index=source_indices, source=T)
        return I

    @nvtx_annotate("norm_mlp")
    def _norm_mlp(self, norm):
        norm = self.init_norm(norm)
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, self.hidden_channels, 3)
        return norm

    @nvtx_annotate("TensorEmbedding")
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        X = self._compute_node_tensor_features(
            z, edge_index, edge_weight, edge_vec, edge_attr
        )  # shape: (n_atoms, hidden_channels, 3, 3)
        # X = self._aggregate_edge_features(
        #     z.shape[0], Xij, edge_index
        # )  # shape: (n_atoms, hidden_channels, 3, 3)
        norm = self._norm_mlp(tensor_norm(X))  # shape: (n_atoms, hidden_channels)
        X = self.linear_tensor(X, norm)  # shape: (n_atoms, hidden_channels, 3, 3)
        return X


@nvtx_annotate("compute_tensor_edge_features")
def compute_tensor_edge_features(X, edge_index, factor):
    I, A, S = decompose_tensor(X)
    msg = factor[..., 1, None, None] * A.index_select(0, edge_index[1]) + factor[
        ..., 2, None, None
    ] * S.index_select(0, edge_index[1])
    msg.diagonal(dim1=-2, dim2=-1).add_(
        factor[..., 0, None] * I.index_select(0, edge_index[1]).unsqueeze(-1)
    )
    return msg


@nvtx_annotate("tensor_message_passing")
def tensor_message_passing(n_atoms: int, edge_index: Tensor, tensor: Tensor) -> Tensor:
    msg = tensor.index_select(
        0, edge_index[1]
    )  # shape = (n_edges, hidden_channels, 3, 3)
    tensor_m = torch.zeros(
        (n_atoms, tensor.shape[1], tensor.shape[2], tensor.shape[3]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    tensor_m = tensor_m.index_add(0, edge_index[0], msg)
    return tensor_m  # shape = (n_atoms, hidden_channels, 3, 3)


class Interaction(nn.Module):
    """Interaction layer.

    :meta private:
    """

    def __init__(
        self,
        num_rbf,
        hidden_channels,
        activation,
        cutoff_lower,
        cutoff_upper,
        equivariance_invariance_group,
        dtype=torch.float32,
    ):
        super(Interaction, self).__init__()

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.tensor_linear_in = TensorLinear(hidden_channels, hidden_channels)
        self.tensor_linear_out = TensorLinear(hidden_channels, hidden_channels)
        self.act = activation()
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.tensor_linear_in.reset_parameters()
        self.tensor_linear_out.reset_parameters()

    @nvtx_annotate("update_tensor_node_features")
    def _update_tensor_node_features(self, X, X_aggregated):
        X = self.tensor_linear_in(X)
        B = torch.matmul(X, X_aggregated)
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(X_aggregated, X)
        elif self.equivariance_invariance_group == "SO(3)":
            A = B
        else:
            raise ValueError("Unknown equivariance group")
        Xnew = A + B
        return Xnew

    @nvtx_annotate("compute_vector_node_features")
    def _compute_vector_node_features(self, edge_attr, edge_weight):
        C = self.cutoff(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.hidden_channels, 3
        )
        return edge_attr

    @nvtx_annotate("Interaction")
    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        q: Tensor,
    ) -> Tensor:
        X = (
            X / (tensor_norm(X) + 1)[..., None, None]
        )  # shape (n_atoms, hidden_channels, 3, 3)
        node_features = self._compute_vector_node_features(
            edge_attr, edge_weight
        )  # shape (n_edges, hidden_channels, 3)
        Y_edges = compute_tensor_edge_features(
            X, edge_index, node_features
        )  # shape (n_edges, hidden_channels, 3, 3)
        Y_aggregated = tensor_message_passing(
            X.shape[0], edge_index, Y_edges
        )  # shape (n_atoms, hidden_channels, 3, 3)
        Xnew = self._update_tensor_node_features(
            X, Y_aggregated
        )  # shape (n_atoms, hidden_channels, 3, 3)
        dX = self.tensor_linear_out(
            Xnew / (tensor_norm(Xnew) + 1)[..., None, None]
        )  # shape (n_atoms, hidden_channels, 3, 3)
        charge_factor = 1 + 0.1 * q[..., None, None, None]
        X = (
            X + (dX + torch.matrix_power(dX, 2)) * charge_factor
        )  # shape (n_atoms, hidden_channels, 3, 3)
        return X
