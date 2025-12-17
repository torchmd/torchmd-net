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
)

__all__ = ["TensorNet"]


def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


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
            trainable_rbf,
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

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        # Obtain graph, with distances and relative position vectors
        edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)
        # This assert convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        # Distance module returns -1 for non-existing edges, to avoid having to resize the tensors when we want to ensure static shapes (for CUDA graphs) we make all non-existing edges pertain to a ghost atom
        # Total charge q is a molecule-wise property. We transform it into an atom-wise property, with all atoms belonging to the same molecule being assigned the same charge q
        if q is None:
            q = torch.zeros_like(z, device=z.device, dtype=z.dtype)
        else:
            q = q[batch]
        zp = z
        if self.static_shapes:
            mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
            zp = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)
            q = torch.cat((q, torch.zeros(1, device=q.device, dtype=q.dtype)), dim=0)
            # I trick the model into thinking that the masked edges pertain to the extra atom
            # WARNING: This can hurt performance if max_num_pairs >> actual_num_pairs
            edge_index = edge_index.masked_fill(mask, z.shape[0])
            edge_weight = edge_weight.masked_fill(mask[0], 0)
            edge_vec = edge_vec.masked_fill(
                mask[0].unsqueeze(-1).expand_as(edge_vec), 0
            )
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] == edge_index[1]
        # Normalizing edge vectors by their length can result in NaNs, breaking Autograd.
        # I avoid dividing by zero by setting the weight of self edges and self loops to 1
        edge_vec = edge_vec / edge_weight.masked_fill(mask, 1).unsqueeze(1)
        X = self.tensor_embedding(zp, edge_index, edge_weight, edge_vec, edge_attr)
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr, q)
        I, A, S = decompose_tensor(X)
        x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(x)
        x = self.act(self.linear((x)))
        # # Remove the extra atom
        if self.static_shapes:
            x = x[:-1]
        return x, None, z, pos, batch


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
        trainable_rbf=False,
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
        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _get_atomic_number_message(self, z: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(z)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
                -1, self.hidden_channels * 2
            )
        )[..., None, None]
        return Zij

    def _get_tensor_messages(
        self, Zij: Tensor, edge_weight: Tensor, edge_vec_norm: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        C = self.cutoff(edge_weight).reshape(-1, 1, 1, 1) * Zij
        eye = torch.eye(3, 3, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)[
            None, None, ...
        ]
        Iij = self.distance_proj1(edge_attr)[..., None, None] * C * eye
        Aij = (
            self.distance_proj2(edge_attr)[..., None, None]
            * C
            * vector_to_skewtensor(edge_vec_norm)[..., None, :, :]
        )
        Sij = (
            self.distance_proj3(edge_attr)[..., None, None]
            * C
            * vector_to_symtensor(edge_vec_norm)[..., None, :, :]
        )
        return Iij, Aij, Sij

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec_norm: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        Zij = self._get_atomic_number_message(z, edge_index)
        Iij, Aij, Sij = self._get_tensor_messages(
            Zij, edge_weight, edge_vec_norm, edge_attr
        )
        source = torch.zeros(
            z.shape[0], self.hidden_channels, 3, 3, device=z.device, dtype=Iij.dtype
        )
        I = source.index_add(dim=0, index=edge_index[0], source=Iij)
        A = source.index_add(dim=0, index=edge_index[0], source=Aij)
        S = source.index_add(dim=0, index=edge_index[0], source=Sij)
        norm = self.init_norm(tensor_norm(I + A + S))
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, self.hidden_channels, 3)
        I = (
            self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 0, None, None]
        )
        A = (
            self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 1, None, None]
        )
        S = (
            self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 2, None, None]
        )
        X = I + A + S
        return X


def tensor_message_passing(
    edge_index: Tensor, factor: Tensor, tensor: Tensor, natoms: int
) -> Tensor:
    """Message passing for tensors."""
    msg = factor * tensor.index_select(0, edge_index[1])
    shape = (natoms, tensor.shape[1], tensor.shape[2], tensor.shape[3])
    tensor_m = torch.zeros(*shape, device=tensor.device, dtype=tensor.dtype)
    tensor_m = tensor_m.index_add(0, edge_index[0], msg)
    return tensor_m


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
        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.act = activation()
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        q: Tensor,
    ) -> Tensor:
        C = self.cutoff(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.hidden_channels, 3
        )
        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = I + A + S
        Im = tensor_message_passing(
            edge_index, edge_attr[..., 0, None, None], I, X.shape[0]
        )
        Am = tensor_message_passing(
            edge_index, edge_attr[..., 1, None, None], A, X.shape[0]
        )
        Sm = tensor_message_passing(
            edge_index, edge_attr[..., 2, None, None], S, X.shape[0]
        )
        msg = Im + Am + Sm
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor((1 + 0.1 * q[..., None, None, None]) * (A + B))
        if self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(2 * B)
        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + (1 + 0.1 * q[..., None, None, None]) * torch.matrix_power(dX, 2)
        return X
