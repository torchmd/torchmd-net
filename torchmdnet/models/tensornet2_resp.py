import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
    MLP,
)
from warnings import warn


from torchmdnet.models.tensornet import (
    _decompose_tensor,
    _compose_tensor,
    _tensor_matmul_o3,
    _tensor_matmul_so3,
    tensor_norm,
    TensorEmbedding,
    tensornet_interaction_message_passing,
)

OPT = True  # This flag controls if the optimized tensornet nvidia warp ops are used TODO: change to set by detection of nvidia warp.
if OPT:
    from torchmdnet.extensions.warp_ops import (
        graph_transform,
        fn_message_passing as fn_tensornet_interaction_message_passing,
        fn_compose_tensor as compose_tensor,
        fn_decompose_tensor as decompose_tensor,
        fn_tensor_matmul_o3_3x3 as tensor_matmul_o3,
        fn_tensor_matmul_so3_3x3 as tensor_matmul_so3,
        fn_tensor_norm3,
    )

else:
    compose_tensor = _compose_tensor
    decompose_tensor = _decompose_tensor
    tensor_matmul_o3 = _tensor_matmul_o3
    tensor_matmul_so3 = _tensor_matmul_so3


__all__ = ["TensorNet2"]


class Qeq(nn.Module):
    def __init__(self, q_dim=16, static_shapes=False):
        super(Qeq, self).__init__()
        self.q_dim = q_dim
        self.static_shapes = static_shapes

    def mol_sum(self, x: Tensor, batch: Tensor) -> Tensor:
        # torch.compile and torch.export don't support .item() calls during tracing
        # The model should be warmed up before compilation to set the correct dim_size
        if torch.compiler.is_compiling():
            pass
        elif torch.jit.is_scripting():
            # TorchScript doesn't support torch.cuda.is_current_stream_capturing()
            # For CPU, always update dim_size (no CUDA graphs on CPU)
            # For CUDA with static_shapes, only update once (first call sets dim_size for CUDA graph capture)
            # For CUDA without static_shapes, always update (dynamic batch sizes)
            if not x.is_cuda or not self.static_shapes or self.dim_size == 0:
                self.dim_size = int(batch.max().item() + 1)
        else:
            is_capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
            if not x.is_cuda or not is_capturing:
                self.dim_size = int(batch.max().item() + 1)
            if is_capturing:
                assert (
                    self.dim_size > 0
                ), "Warming up is needed before capturing the model into a CUDA graph"
                warn(
                    "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                        self.dim_size
                    )
                )

        if self.dim_size == 1:
            # do the direct sum
            out = x.sum(dim=0).unsqueeze(0)
        else:
            out = torch.zeros(
                self.dim_size, x.shape[-1], device=x.device, dtype=x.dtype
            )
            out = out.index_add(0, batch, x)

        return out

    def forward(self, old_charges, f, batch, Q) -> Tensor:

        # if we are using static shapes we can skip the last dummy atom
        if self.static_shapes and not OPT:
            f = f[:-1]
            old_charges = old_charges[:-1]
            Q = Q[:-1]  # Q is already per atom

        f_u = f**2

        epsilon = 1.0e-6

        F_u = self.mol_sum(f_u, batch)
        F_u = F_u + epsilon
        Q_u = self.mol_sum(old_charges, batch)

        # Q is already per atom
        dQ = Q.unsqueeze(-1) - Q_u[batch]

        F_u = F_u[batch]

        _f = f_u / F_u
        new_charges = old_charges + _f * dQ

        if self.static_shapes and not OPT:  # add back the dummy atom so shapes match
            new_charges = torch.cat(
                (
                    new_charges,
                    torch.zeros(
                        (1, new_charges.shape[1]),
                        device=new_charges.device,
                        dtype=new_charges.dtype,
                    ),
                )
            )

        return new_charges


class ChargePredict(nn.Module):
    def __init__(
        self,
        hidden_channels,
        activation,
        q_dim=16,
    ):
        super(ChargePredict, self).__init__()
        self.q_dim = q_dim
        self.q_norm = nn.LayerNorm(hidden_channels)
        self.q_mlp = MLP(
            hidden_channels,
            2 * q_dim,
            hidden_channels,
            activation,
            1,
        )

    def reset_parameters(self):
        self.q_norm.reset_parameters()
        self.q_mlp.reset_parameters()

    def forward(self, X, batch):
        _x = tensor_norm(X)
        _charges_f = self.q_mlp(self.q_norm(_x))
        charges, _f = _charges_f[:, : self.q_dim], _charges_f[:, self.q_dim :]
        return charges, _f


class TensorNet2(nn.Module):
    r"""TensorNet with Neutral Charge Equilibration
    
    
    Combines AIMNet2 style neutral charge equilibration procedure with TensorNet
    
    
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
        q_dim (int, optional): Charge channel size.
            (default: :obj:`16`)
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
        output_charges (bool, optional): Whether charges should be output. The current 
            output is done by appending to the node features x that will be passed to the output 
            model. If True the output model must be compatible with the output charges.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels=128,
        q_dim=16,
        mp_dim=128,
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
        output_charges=False,
    ):
        super(TensorNet2, self).__init__()

        self.q_dim = q_dim
        self.output_charges = output_charges

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

        if isinstance(mp_dim, int):
            if mp_dim > hidden_channels:
                Warning(
                    "mp_dim cannot be larger than hidden_channels, setting mp_dim = hidden_channels"
                )
                mp_dim = hidden_channels
            self.mp_dims = [mp_dim] * num_layers
        elif isinstance(mp_dim, list):
            if len(mp_dim) != num_layers:
                raise ValueError("mp_dim must be a list of length num_layers")
            self.mp_dims = mp_dim
        else:
            raise ValueError("mp_dim must be an int or a list of ints")

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

        self.charge_predict_0 = ChargePredict(
            hidden_channels,
            activation,
            self.q_dim,
        )
        self.qeq = Qeq(self.q_dim, static_shapes)

        self.layers = nn.ModuleList()
        self.charge_predicts = nn.ModuleList()
        if num_layers != 0:
            for l in range(num_layers):
                _layer = Interaction(
                    num_rbf,
                    hidden_channels,
                    self.mp_dims[l],
                    self.q_dim,
                    activation,
                    cutoff_lower,
                    cutoff_upper,
                    equivariance_invariance_group,
                    static_shapes,
                    dtype,
                )
                self.layers.append(_layer)
                _charge_predict = ChargePredict(
                    hidden_channels,
                    activation,
                    self.q_dim,
                )
                self.charge_predicts.append(_charge_predict)

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
        self.charge_predict_0.reset_parameters()
        for layer in self.charge_predicts:
            layer.reset_parameters()

    def setup_for_inference(self, z, batch):
        self.tensor_embedding.setup_for_inference(z, batch)
        self.inference_mode = True

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

        Q = q  # total molecule charge

        if OPT:
            # perpare graph indices for message passing
            row_data, row_indices, row_indptr, col_data, col_indices, col_indptr = (
                graph_transform(edge_index.int(), z.shape[0])
            )
        else:
            row_data, row_indices, row_indptr, col_data, col_indices, col_indptr = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # This assert convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        # Total charge Q is a molecule-wise property. We transform it into an atom-wise property, with all atoms belonging to the same molecule being assigned the same charge q

        if Q is not None:
            Q = Q[batch]
        else:
            Q = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            warn("No charges provided, setting to zero")

        zp = z
        if self.static_shapes:
            mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
            zp = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)

            if not OPT:
                Q = torch.cat(
                    (Q, torch.zeros(1, device=Q.device, dtype=Q.dtype)), dim=0
                )
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

        graph_info = {
            "edge_index": edge_index,
            "row_data": row_data,
            "row_indices": row_indices,
            "row_indptr": row_indptr,
            "col_data": col_data,
            "col_indices": col_indices,
            "col_indptr": col_indptr,
        }

        X = self.tensor_embedding(zp, graph_info, edge_weight, edge_vec, edge_attr)

        # from the initial embedding we compute some partial charges
        charges, _f_0 = self.charge_predict_0(X, batch)
        charges = self.qeq(charges, _f_0, batch, Q)

        for l, (layer, charge_predictor) in enumerate(
            zip(self.layers, self.charge_predicts)
        ):
            X = X / (tensor_norm(X) + 1)[:, None, None, :]

            # update node tensor features
            dX = layer(X, charges, graph_info, edge_weight, edge_attr, edge_vec)
            X = X + dX

            # predict new charges (residual update)
            dcharges, _f = charge_predictor(dX, batch)
            charges = charges + dcharges
            charges = self.qeq(charges, _f, batch, Q)

        if not OPT:
            I, A, S = decompose_tensor(X)
            x = torch.cat((3 * I**2, tensor_norm(A), tensor_norm(S)), dim=-1)
            x = self.out_norm(x)
            x = self.act(self.linear((x)))

            # append charges for coulomb in output module
            if self.output_charges:
                x = torch.cat([x, charges], dim=-1)

            # Remove the extra atom
            if self.static_shapes:
                x = x[:-1]

        else:

            x = fn_tensor_norm3(X)
            x = self.out_norm(x)
            x = self.act(self.linear((x)))

            # append charges for coulomb in output module
            if self.output_charges:
                x = torch.cat([x, charges], dim=-1)

        return x, None, z, pos, batch


class Interaction(nn.Module):
    """Interaction layer.

    :meta private:
    """

    def __init__(
        self,
        num_rbf,
        hidden_channels,
        mp_dim,
        q_dim,
        activation,
        cutoff_lower,
        cutoff_upper,
        equivariance_invariance_group,
        static_shapes,
        dtype=torch.float32,
    ):
        super(Interaction, self).__init__()

        assert (
            mp_dim <= hidden_channels
        ), "mp_dim must be less than or equal to hidden_channels"

        self.hidden_channels = hidden_channels
        self.mp_dim = mp_dim
        self.q_dim = q_dim
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.static_shapes = static_shapes
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(num_rbf + 2 * q_dim, self.mp_dim, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(self.mp_dim, 2 * self.mp_dim, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(
                2 * self.mp_dim,
                3 * self.mp_dim,
                bias=True,
                dtype=dtype,
            )
        )
        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(self.hidden_channels, self.mp_dim, bias=False)
            )
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(
                    self.mp_dim,
                    self.hidden_channels,
                    bias=False,
                )
            )

        self.act = act_class_mapping[activation]()
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
        charges: Tensor,
        graph_info: dict[str, Tensor],
        edge_weight: Tensor,
        edge_attr: Tensor,
        edge_vec_norm: Tensor,
    ) -> Tensor:

        C = self.cutoff(edge_weight)

        edge_index = graph_info["edge_index"]

        # add the partial charges as edge features and then do a normal tensornet update

        if OPT and self.static_shapes:
            # add a dummy atom to the charges so we can do index select correctly with static shapes
            _charges = torch.cat(
                (
                    charges,
                    torch.zeros(
                        1, charges.shape[1], device=charges.device, dtype=charges.dtype
                    ),
                ),
                dim=0,
            )

            charge_edges_i = _charges.index_select(0, edge_index[0])
            charge_edges_j = _charges.index_select(0, edge_index[1])
        else:
            charge_edges_i = charges.index_select(0, edge_index[0])
            charge_edges_j = charges.index_select(0, edge_index[1])

        x_edge_attr = torch.cat([edge_attr, charge_edges_i, charge_edges_j], dim=-1)

        for linear_scalar in self.linears_scalar:
            x_edge_attr = self.act(linear_scalar(x_edge_attr))
        x_edge_attr = (x_edge_attr * C.view(-1, 1)).reshape(
            x_edge_attr.shape[0], 3, self.mp_dim
        )

        I, A, S = decompose_tensor(X)

        I = self.linears_tensor[0](I)
        A = self.linears_tensor[1](A)
        S = self.linears_tensor[2](S)
        Y = compose_tensor(I, A, S)

        if not OPT:
            edge_index = graph_info["edge_index"]

            Im, Am, Sm = tensornet_interaction_message_passing(
                I, A, S, x_edge_attr, edge_index, X.shape[0]
            )

        else:

            row_data = graph_info["row_data"]
            row_indices = graph_info["row_indices"]
            row_indptr = graph_info["row_indptr"]
            col_data = graph_info["col_data"]
            col_indices = graph_info["col_indices"]
            col_indptr = graph_info["col_indptr"]

            Im, Am, Sm = fn_tensornet_interaction_message_passing(
                I,
                A,
                S,
                x_edge_attr,
                row_data,
                row_indices,
                row_indptr,
                col_data,
                col_indices,
                col_indptr,
            )

        msg = compose_tensor(Im, Am, Sm)

        if self.equivariance_invariance_group == "O(3)":
            C = tensor_matmul_o3(Y, msg)
            I, A, S = decompose_tensor(C)
        if self.equivariance_invariance_group == "SO(3)":
            C = 2 * tensor_matmul_so3(Y, msg)
            I, A, S = decompose_tensor(C)

        normp1 = tensor_norm(C) + 1

        if not OPT:
            I, A, S = (
                I / normp1,
                A / normp1[..., None, None, :],
                S / normp1[..., None, None, :],
            )
        else:
            I = I / normp1.unsqueeze(1)
            A = A / normp1.unsqueeze(1)
            S = S / normp1.unsqueeze(1)

        I = self.linears_tensor[3](I)
        A = self.linears_tensor[4](A)
        S = self.linears_tensor[5](S)
        dX = compose_tensor(I, A, S)

        return dX + tensor_matmul_so3(dX, dX)
