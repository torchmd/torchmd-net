# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchmdnet.extensions.ops import get_neighbor_pairs_kernel
import warnings
from ase.data import covalent_radii


def visualize_basis(basis_type, num_rbf=50, cutoff_lower=0, cutoff_upper=5):
    """
    Function for quickly visualizing a specific basis. This is useful for inspecting
    the distance coverage of basis functions for non-default lower and upper cutoffs.

    Args:
        basis_type (str): Specifies the type of basis functions used. Can be one of
            ['gauss',expnorm']
        num_rbf (int, optional): The number of basis functions.
            (default: :obj:`50`)
        cutoff_lower (float, optional): The lower cutoff of the basis.
            (default: :obj:`0`)
        cutoff_upper (float, optional): The upper cutoff of the basis.
            (default: :obj:`5`)
    """
    import matplotlib.pyplot as plt

    distances = torch.linspace(cutoff_lower - 1, cutoff_upper + 1, 1000)
    basis_kwargs = {
        "num_rbf": num_rbf,
        "cutoff_lower": cutoff_lower,
        "cutoff_upper": cutoff_upper,
    }
    basis_expansion = rbf_class_mapping[basis_type](**basis_kwargs)
    expanded_distances = basis_expansion(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()


class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        cutoff_lower,
        cutoff_upper,
        max_z=100,
        dtype=torch.float32,
    ):
        """
        The ET architecture assigns two  learned vectors to each atom type
        zi. One  is used to  encode information  specific to an  atom, the
        other (this  class) takes  the role  of a  neighborhood embedding.
        The neighborhood embedding, which is  an embedding of the types of
        neighboring atoms, is multiplied by a distance filter.


        This embedding allows  the network to store  information about the
        interaction of atom pairs.

        See eq. 3 in https://arxiv.org/pdf/2202.02541.pdf for more details.
        """
        super(NeighborEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Args:
            z (Tensor): Atomic numbers of shape :obj:`[num_nodes]`
            x (Tensor): Node feature matrix (atom positions) of shape :obj:`[num_nodes, 3]`
            edge_index (Tensor): Graph connectivity (list of neighbor pairs) with shape :obj:`[2, num_edges]`
            edge_weight (Tensor): Edge weight vector of shape :obj:`[num_edges]`
            edge_attr (Tensor): Edge attribute matrix of shape :obj:`[num_edges, 3]`
        Returns:
            x_neighbors (Tensor): The embedding of the neighbors of each atom of shape :obj:`[num_nodes, hidden_channels]`
        """
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        msg = W * x_neighbors.index_select(0, edge_index[1])
        x_neighbors = torch.zeros(
            z.shape[0], x.shape[1], dtype=x.dtype, device=x.device
        ).index_add(0, edge_index[0], msg)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors


class OptimizedDistance(torch.nn.Module):
    """Compute the neighbor list for a given cutoff.

    This operation can be placed inside a CUDA graph in some cases.
    In particular, resize_to_fit must be False.

    Note that this module returns neighbors such that :math:`r_{ij} \\ge \\text{cutoff_lower}\\quad\\text{and}\\quad r_{ij} < \\text{cutoff_upper}`.

    This function optionally supports periodic boundary conditions with
    arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
    certain requirements:

    .. code:: python

       a[1] = a[2] = b[2] = 0
       a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
       a[0] >= 2*b[0]
       a[0] >= 2*c[0]
       b[1] >= 2*c[1]

    These requirements correspond to a particular rotation of the system and
    reduced form of the vectors, as well as the requirement that the cutoff be
    no larger than half the box width.

    Parameters
    ----------
    cutoff_lower : float
        Lower cutoff for the neighbor list.
    cutoff_upper : float
        Upper cutoff for the neighbor list.
    max_num_pairs : int
        Maximum number of pairs to store, if the number of pairs found is less than this, the list is padded with (-1,-1) pairs up to max_num_pairs unless resize_to_fit is True, in which case the list is resized to the actual number of pairs found.
        If the number of pairs found is larger than this, an asynchronous device-side error is raised.
        If negative, it is interpreted as (minus) the maximum number of neighbors per atom.
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["brute", "cell"]`.

        1. *Brute*: A brute force O(N^2) algorithm, best for small number of particles.
        2. *Cell*:  A cell list algorithm, best for large number of particles, low cutoffs and low batch size.
    box : torch.Tensor, optional
        The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a ox per sample is desired.
        where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
        If this is omitted, periodic boundary conditions are not applied.
    loop : bool, optional
        Whether to include self-interactions.
        Default: False
    include_transpose : bool, optional
        Whether to include the transpose of the neighbor list.
        Default: True
    resize_to_fit : bool, optional
        Whether to resize the neighbor list to the actual number of pairs found. When False, the list is padded with (-1,-1) pairs up to max_num_pairs
        Default: True
        If this is True the operation is not CUDA graph compatible.
    return_vecs : bool, optional
        Whether to return the distance vectors.
        Default: False
    long_edge_index : bool, optional
        Whether to return edge_index as int64, otherwise int32.
        Default: True
    """

    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=-32,
        return_vecs=False,
        loop=False,
        strategy="brute",
        include_transpose=True,
        resize_to_fit=True,
        box=None,
        long_edge_index=True,
    ):
        super(OptimizedDistance, self).__init__()
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower
        self.max_num_pairs = max_num_pairs
        self.strategy = strategy
        self.loop = loop
        self.return_vecs = return_vecs
        self.include_transpose = include_transpose
        self.resize_to_fit = resize_to_fit
        self.use_periodic = True
        self.num_cells = 0

        if box is None:
            self.use_periodic = False
            if strategy == "cell":
                lbox = cutoff_upper * 3.0
                box = torch.tensor(
                    [[lbox, 0, 0], [0, lbox, 0], [0, 0, lbox]], device="cpu"
                )
            else:
                box = torch.zeros((3, 3), device="cpu")

        self.register_buffer("box", box, persistent=True)

        if self.strategy == "cell":
            from torchmdnet.extensions.neighbor_utils import get_cell_dimensions

            cell_dims = get_cell_dimensions(
                self.box[0, 0], self.box[1, 1], self.box[2, 2], cutoff_upper
            )
            self.num_cells = int(cell_dims.prod())
            if self.num_cells > 1024**3:
                raise RuntimeError(
                    f"Too many cells: {self.num_cells}. Maximum is 1024^3. "
                    f"Reduce box size or increase cutoff."
                )

        self.long_edge_index = long_edge_index

    def forward(
        self, pos: Tensor, batch: Optional[Tensor] = None, box: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute the neighbor list for a given cutoff.

        Parameters
        ----------
        pos : torch.Tensor
            A tensor with shape (N, 3) representing the positions.
        batch : torch.Tensor, optional
            A tensor with shape (N,). Defaults to None.
        box : torch.Tensor, optional
            The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)`,
        Returns
        -------
        edge_index : torch.Tensor
            List of neighbors for each atom in the batch.
            Shape is (2, num_found_pairs) or (2, max_num_pairs).
        edge_weight : torch.Tensor
            List of distances for each atom in the batch.
            Shape is (num_found_pairs,) or (max_num_pairs,).
        edge_vec : torch.Tensor, optional
            List of distance vectors for each atom in the batch.
            Shape is (num_found_pairs, 3) or (max_num_pairs, 3).

        Notes
        -----
        If `resize_to_fit` is True, the tensors will be trimmed to the actual number of pairs found.
        Otherwise, the tensors will have size `max_num_pairs`, with neighbor pairs (-1, -1) at the end.
        """
        use_periodic = self.use_periodic
        if not use_periodic:
            use_periodic = box is not None

        # Use the registered buffer box if no box is provided
        # The buffer is automatically moved to the correct device with the module
        if box is None:
            box = self.box

        assert box is not None, "Box must be provided"

        # Ensure box has correct dtype (device is already correct if using self.box)
        if box.dtype != pos.dtype:
            box = box.to(dtype=pos.dtype)

        max_pairs = self.max_num_pairs
        if self.max_num_pairs < 0:
            max_pairs = -self.max_num_pairs * pos.shape[0]
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs_kernel(
            strategy=self.strategy,
            positions=pos,
            batch=batch,
            max_num_pairs=max_pairs,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            loop=self.loop,
            include_transpose=self.include_transpose,
            box_vectors=box,
            use_periodic=use_periodic,
            num_cells=self.num_cells,
        )
        torch._assert_async(
            (num_pairs[0] <= max_pairs),
            "Found num_pairs > max_num_pairs, please increase max_num_pairs",
        )

        # Remove (-1,-1)  pairs
        if self.resize_to_fit:
            mask = edge_index[0] != -1
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_vec = edge_vec[mask, :]
        if self.long_edge_index:
            edge_index = edge_index.to(torch.long)
        if self.return_vecs:
            return edge_index, edge_weight, edge_vec
        else:
            return edge_index, edge_weight, None


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, dtype=self.dtype
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class GLU(nn.Module):
    r"""Applies the gated linear unit (GLU) function:

    .. math::

        \text{GLU}(x) = \text{Linear}_1(x) \otimes \sigma(\text{Linear}_2(x))


    where :math:`\otimes` is the element-wise multiplication operator and
    :math:`\sigma` is an activation function.

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int, optional): Number of hidden features. Defaults to None, meaning hidden_channels=in_channels.
        activation (nn.Module, optional): Activation function to use. Defaults to Sigmoid.
    """

    def __init__(
        self, in_channels, hidden_channels=None, activation: Optional[nn.Module] = None
    ):
        super(GLU, self).__init__()
        self.act = nn.Sigmoid() if activation is None else activation
        hidden_channels = hidden_channels or in_channels
        self.W = nn.Linear(in_channels, hidden_channels)
        self.V = nn.Linear(in_channels, hidden_channels)

    def forward(self, x):
        return self.W(x) * self.act(self.V(x))


class ShiftedSoftplus(nn.Module):
    r"""Applies the ShiftedSoftplus function :math:`\text{ShiftedSoftplus}(x) = \frac{1}{\beta} *
    \log(1 + \exp(\beta * x))-\log(2)` element-wise.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    """Swish activation function as defined in https://arxiv.org/pdf/1710.05941 :

    .. math::

        \text{Swish}(x) = x \cdot \sigma(\beta x)

    Args:
        beta (float, optional): Scaling factor for Swish activation. Defaults to 1.

    """

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    """SwiGLU activation function as defined in https://arxiv.org/pdf/2002.05202 :

    .. math::

        \text{SwiGLU}(x) = \text{Linear}_1(x) \otimes \text{Swish}(\text{Linear}_2(x))

    W1, V have shape (in_features, hidden_features)
    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to None, meaning hidden_features=in_features.
        beta (float, optional): Scaling factor for Swish activation. Defaults to 1.0.
    """

    def __init__(self, in_features, hidden_features=None, beta=1.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        act = Swish(beta)
        self.glu = GLU(in_features, hidden_features, activation=act)

    def forward(self, x):
        return self.glu(x)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs


class MLP(nn.Module):
    r"""A simple multi-layer perceptron with a given number of layers and hidden channels.

    The simplest MLP has no hidden layers and is composed of two linear layers with a non-linear activation function in between:

    .. math::

        \text{MLP}(x) = \text{Linear}_o(\text{act}(\text{Linear}_i(x)))

    Where :math:`\text{Linear}_i` has input size :math:`\text{in_channels}` and output size :math:`\text{hidden_channels}` and :math:`\text{Linear}_o` has input size :math:`\text{hidden_channels}` and output size :math:`\text{out_channels}`.


    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_channels (int): Number of hidden features.
        activation (str): Activation function to use.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to 0.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float32.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        activation,
        num_hidden_layers=0,
        dtype=torch.float32,
    ):
        super(MLP, self).__init__()
        act_class = act_class_mapping[activation]
        self.act = act_class()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(in_channels, hidden_channels, dtype=dtype))
        self.layers.append(self.act)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels, dtype=dtype))
            self.layers.append(self.act)
        self.layers.append(nn.Linear(hidden_channels, out_channels, dtype=dtype))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.layers(x)
        return x


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
        dtype=torch.float,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.vec2_proj = nn.Linear(
            hidden_channels, out_channels, bias=False, dtype=dtype
        )

        act_class = act_class_mapping[activation]
        self.update_net = MLP(
            in_channels=hidden_channels * 2,
            out_channels=out_channels * 2,
            hidden_channels=intermediate_channels,
            activation=activation,
            num_hidden_layers=0,
            dtype=dtype,
        )
        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        self.update_net.reset_parameters()

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0),
            vec1_buffer.size(2),
            device=vec1_buffer.device,
            dtype=vec1_buffer.dtype,
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        if not mask.all():
            warnings.warn(
                (
                    f"Skipping gradients for {(~mask).sum()} atoms due to vector features being zero. "
                    "This is likely due to atoms being outside the cutoff radius of any other atom. "
                    "These atoms will not interact with any other atom unless you change the cutoff."
                )
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Broadcasts src to the shape of other along the given dimension."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """Has the signature of torch_scatter.scatter, but uses torch.scatter_reduce instead."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    operation_dict = {
        "add": "sum",
        "sum": "sum",
        "mul": "prod",
        "mean": "mean",
        "min": "amin",
        "max": "amax",
    }
    reduce_op = operation_dict[reduce]
    # take into account the dimensionality of src
    index = _broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    res = out.scatter_reduce(dim, index, src, reduce_op)
    return res


class RepulsionZBL(nn.Module):
    r"""Ziegler-Biersack-Littmark screened nuclear repulsion, computed on the model's own
    short-range neighbour list.

    .. math::
        E_\mathrm{rep} = \tfrac12 \sum_{i \ne j} k_e \frac{Z_a Z_b}{r}\, \phi(r/a)\, f_c(r)

        \phi(x) = 0.1818 e^{-3.2x} + 0.5099 e^{-0.9423x} + 0.2802 e^{-0.4029x} + 0.02817 e^{-0.2016x}

        a = \frac{0.4543\, a_0}{Z_a^{0.3} + Z_b^{0.3}}

    MODIFIED FROM https://github.com/orbital-materials/orb-models/blob/main/orb_models/forcefield/models/pair_repulsion.py

    Apache License Version 2.0, January 2004 http://www.apache.org/licenses/

    """

    ANGSTROM_TO_BOHR = 1.8897261258369282
    COULOMB_EV_ANGSTROM = 14.399645  # e^2 / (4 pi eps0), in eV*Angstrom
    A_PREFACTOR = 0.4543
    A_EXP = 0.300
    ENVELOPE_P = 6

    def __init__(self, max_z: int = 87):
        super().__init__()
        num_elem = min(max_z, len(covalent_radii))
        z = torch.arange(num_elem, dtype=torch.get_default_dtype())
        z[0] = 1  # avoid 0**A_EXP for the unused index-0 slot; never selected (atomic numbers >= 1)
        cov_radius = torch.tensor(covalent_radii[:num_elem], dtype=torch.get_default_dtype())
        # Persistent on purpose, even though every one of these is a fixed table derivable
        # from the constants above. Checkpoints trained with repulsion=True fold this term into
        # the total energy they were fit against, but older torchmd-net versions predate the
        # `repulsion` hyperparameter and silently ignore it -- they would build a plain
        # TensorNet2, load such a checkpoint without complaint, and return energies missing the
        # ZBL term entirely (wrong by ~kcal/mol, with nothing in the output to say so). Keeping
        # these tables in the state_dict makes that failure loud instead: the strict
        # load_state_dict() in load_model() reports them as unexpected keys. Secondary benefit:
        # the exact tables the model was trained with travel with the weights.
        self.register_buffer("z_pow", z**self.A_EXP)
        self.register_buffer("z_outer", torch.outer(z, z))
        self.register_buffer(
            "cutoff_ab", cov_radius.unsqueeze(0) + cov_radius.unsqueeze(1)
        )
        self.num_elem = int(num_elem)

    def forward(self, z: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # per-atom repulsion energy [num_z], reusing the model's own neighbour list
        mask = edge_weight > 0  # drop self-loops (d=0) and padded edges
        r = edge_weight.masked_fill(~mask, 1.0)  # Angstrom
        za = z[edge_index[0]].clamp(max=self.num_elem - 1)
        zb = z[edge_index[1]].clamp(max=self.num_elem - 1)

        screening_length = self.A_PREFACTOR * (1.0 / self.ANGSTROM_TO_BOHR) / (
            self.z_pow[za] + self.z_pow[zb]
        )
        x = r / screening_length
        phi = (
            0.1818 * torch.exp(-3.2 * x)
            + 0.5099 * torch.exp(-0.9423 * x)
            + 0.2802 * torch.exp(-0.4029 * x)
            + 0.02817 * torch.exp(-0.2016 * x)
        )

        cutoff = self.cutoff_ab[za, zb].to(r.dtype)
        p = float(self.ENVELOPE_P)
        rr = (r / cutoff).clamp(max=1.0)
        fc = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * rr.pow(p)
            + p * (p + 2.0) * rr.pow(p + 1.0)
            - (p * (p + 1.0) / 2.0) * rr.pow(p + 2.0)
        ) * (r < cutoff)

        z_ab = self.z_outer[za, zb].to(r.dtype)
        e_ij = self.COULOMB_EV_ANGSTROM * (z_ab / r) * phi * fc
        e_ij = e_ij * mask.to(e_ij.dtype)
        e_atom = torch.zeros(z.shape[0], dtype=e_ij.dtype, device=e_ij.device)
        e_atom = e_atom.index_add(0, edge_index[0], e_ij)
        return 0.5 * e_atom  # half: each pair counted twice


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
    "mish": nn.Mish,
}

dtype_mapping = {16: torch.float16, 32: torch.float, 64: torch.float64}
