import torch
import numpy as np
from typing import Optional, Tuple
from torch import Tensor, nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torchmdnet.models.utils import (
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)

# Creates a skew-symmetric tensor from a vector
def vector_to_skewtensor(vector):
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


# Creates a symmetric traceless tensor from the outer product of a vector with itself
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


# Full tensor decomposition into irreducible components
def decompose_tensor(tensor):
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


# Modifies tensor by multiplying invariant features to irreducible components
def new_radial_tensor(I, A, S, f_I, f_A, f_S):
    I = f_I[..., None, None] * I
    A = f_A[..., None, None] * A
    S = f_S[..., None, None] * S
    return I, A, S


# Computes Frobenius norm
def tensor_norm(tensor):
    return (tensor**2).sum((-2, -1))


class TensorNet(nn.Module):
    r"""TensorNet's architecture, from TensorNet: Cartesian Tensor Representations
        for Efficient Learning of Molecular Potentials; G. Simeon and G. de Fabritiis.

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
        max_z=128,
        max_num_neighbors=64,
        equivariance_invariance_group="O(3)",
        dtype=torch.float32,
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
        self.distance = Distance(
            cutoff_lower, cutoff_upper, max_num_neighbors, return_vecs=True, loop=True
        )
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
        ).jittable()

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
                    ).jittable()
                )
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)
        self.out_norm = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
        self.act = act_class()
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
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        # Obtain graph, with distances and relative position vectors
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        # This assert convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        # Expand distances with radial basis functions
        edge_attr = self.distance_expansion(edge_weight)
        # Embedding from edge-wise tensors to node-wise tensors
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
        # Interaction layers
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr)
        I, A, S = decompose_tensor(X)
        x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(x)
        x = self.act(self.linear((x)))
        return x, None, z, pos, batch


class TensorEmbedding(MessagePassing):
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
        super(TensorEmbedding, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = torch.nn.Embedding(max_z, hidden_channels, dtype=dtype)
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

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec: Tensor,
        edge_attr: Tensor,
    ):

        Z = self.emb(z)
        C = self.cutoff(edge_weight)
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        Iij, Aij, Sij = new_radial_tensor(
            torch.eye(3, 3, device=edge_vec.device, dtype=edge_vec.dtype)[
                None, None, :, :
            ],
            vector_to_skewtensor(edge_vec)[..., None, :, :],
            vector_to_symtensor(edge_vec)[..., None, :, :],
            W1,
            W2,
            W3,
        )
        # propagate_type: (Z: Tensor, I: Tensor, A: Tensor, S: Tensor)
        I, A, S = self.propagate(edge_index, Z=Z, I=Iij, A=Aij, S=Sij, size=None)
        norm = tensor_norm(I + A + S)
        norm = self.init_norm(norm)
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(norm.shape[0], self.hidden_channels, 3)
        I, A, S = new_radial_tensor(I, A, S, norm[..., 0], norm[..., 1], norm[..., 2])
        X = I + A + S

        return X

    def message(self, Z_i, Z_j, I, A, S):
        zij = torch.cat((Z_i, Z_j), dim=-1)
        Zij = self.emb2(zij)
        I = Zij[..., None, None] * I
        A = Zij[..., None, None] * A
        S = Zij[..., None, None] * S

        return I, A, S

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        I, A, S = features
        I = scatter(I, index, dim=self.node_dim, dim_size=dim_size)
        A = scatter(A, index, dim=self.node_dim, dim_size=dim_size)
        S = scatter(S, index, dim=self.node_dim, dim_size=dim_size)

        return I, A, S

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return inputs


class Interaction(MessagePassing):
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
        super(Interaction, self).__init__(aggr="add", node_dim=0)

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype))
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

    def forward(self, X, edge_index, edge_weight, edge_attr):

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
        # propagate_type: (I: Tensor, A: Tensor, S: Tensor, edge_attr: Tensor)
        Im, Am, Sm = self.propagate(
            edge_index, I=I, A=A, S=S, edge_attr=edge_attr, size=None
        )
        msg = Im + Am + Sm
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(A + B)
        if self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(2 * B)
        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + torch.matmul(dX,dX)
        return X

    def message(self, I_j, A_j, S_j, edge_attr):

        I, A, S = new_radial_tensor(
            I_j, A_j, S_j, edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
        )
        return I, A, S

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        I, A, S = features
        I = scatter(I, index, dim=self.node_dim, dim_size=dim_size)
        A = scatter(A, index, dim=self.node_dim, dim_size=dim_size)
        S = scatter(S, index, dim=self.node_dim, dim_size=dim_size)
        return I, A, S

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return inputs
