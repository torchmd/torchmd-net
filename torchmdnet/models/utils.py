# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchmdnet.extensions import get_neighbor_pairs_kernel
import warnings


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
    In particular, resize_to_fit and check_errors must be False.

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
        If the number of pairs found is larger than this, the pairs are randomly sampled. When check_errors is True, an exception is raised in this case.
        If negative, it is interpreted as (minus) the maximum number of neighbors per atom.
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.

        1. *Shared*: An O(N^2) algorithm that leverages CUDA shared memory, best for large number of particles.
        2. *Brute*: A brute force O(N^2) algorithm, best for small number of particles.
        3. *Cell*:  A cell list algorithm, best for large number of particles, low cutoffs and low batch size.
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
    check_errors : bool, optional
        Whether to check for too many pairs. If this is True the operation is not CUDA graph compatible.
        Default: True
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
        check_errors=True,
        box=None,
        long_edge_index=True,
    ):
        super(OptimizedDistance, self).__init__()
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower
        self.max_num_pairs = max_num_pairs
        self.strategy = strategy
        self.box: Optional[Tensor] = box
        self.loop = loop
        self.return_vecs = return_vecs
        self.include_transpose = include_transpose
        self.resize_to_fit = resize_to_fit
        self.use_periodic = True
        if self.box is None:
            self.use_periodic = False
            self.box = torch.empty((0, 0))
            if self.strategy == "cell":
                # Default the box to 3 times the cutoff, really inefficient for the cell list
                lbox = cutoff_upper * 3.0
                self.box = torch.tensor(
                    [[lbox, 0, 0], [0, lbox, 0], [0, 0, lbox]], device="cpu"
                )
        if self.strategy == "cell":
            self.box = self.box.cpu()
        self.check_errors = check_errors
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
        box = self.box if box is None else box
        assert box is not None, "Box must be provided"
        box = box.to(pos.dtype)
        max_pairs: int = self.max_num_pairs
        if self.max_num_pairs < 0:
            max_pairs = -self.max_num_pairs * pos.shape[0]
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs_kernel(
            strategy=self.strategy,
            positions=pos,
            batch=batch,
            max_num_pairs=int(max_pairs),
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            loop=self.loop,
            include_transpose=self.include_transpose,
            box_vectors=box,
            use_periodic=use_periodic,
        )
        if self.check_errors:
            assert (
                num_pairs[0] <= max_pairs
            ), f"Found num_pairs({num_pairs[0]}) > max_num_pairs({max_pairs})"

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
