from pytest import mark, raises
import torch
from torch.autograd import grad
from torchmdnet.models.model import create_model
from torchmdnet.models.utils import Distance
from utils import load_example_args


@mark.parametrize("cutoff_lower", [0, 2])
@mark.parametrize("cutoff_upper", [5, 10])
@mark.parametrize("return_vecs", [False, True])
@mark.parametrize("loop", [False, True])
def test_distance_calculation(cutoff_lower, cutoff_upper, return_vecs, loop):
    dist = Distance(
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=100,
        return_vecs=return_vecs,
        loop=loop,
    )

    batch = torch.tensor([0, 0])

    loop_extra = len(batch) if loop else 0

    # two atoms, distance between lower and upper cutoff
    pos = torch.tensor(
        [[0, 0, 0], [(cutoff_lower + cutoff_upper) / 2, 0, 0]], dtype=torch.float
    )
    edge_index, edge_weight, edge_vec = dist(pos, batch)
    assert (
        edge_index.size(1) == 2 + loop_extra
    ), "Didn't return right number of neighbors"

    # check return_vecs
    if return_vecs:
        assert (
            edge_vec is not None
        ), "Edge vectors were requested but Distance returned None"

    # two atoms, distance lower than lower cutoff
    if cutoff_lower > 0:
        pos = torch.tensor([[0, 0, 0], [cutoff_lower / 2, 0, 0]], dtype=torch.float)
        edge_index, edge_weight, edge_vec = dist(pos, batch)
        assert edge_index.size(1) == loop_extra, "Returned too many neighbors"

    # two atoms, distance larger than upper cutoff
    pos = torch.tensor([[0, 0, 0], [cutoff_upper + 1, 0, 0]], dtype=torch.float)
    edge_index, edge_weight, edge_vec = dist(pos, batch)
    assert edge_index.size(1) == loop_extra, "Returned too many neighbors"

    # check large number of atoms
    batch = torch.zeros(100, dtype=torch.long)
    pos = torch.rand(100, 3)
    edge_index, edge_weight, edge_vec = dist(pos, batch)

    loop_extra = len(batch) if loop else 0

    if cutoff_lower > 0:
        assert edge_index.size(1) == loop_extra, "Expected only self loops"
    else:
        assert edge_index.size(1) == (
            len(batch) * (len(batch) - 1) + loop_extra
        ), "Expected all neighbors to match"


def test_neighbor_count_error():
    dist = Distance(0, 5, max_num_neighbors=32)

    # single molecule that should produce an error due to exceeding
    # the maximum number of neighbors
    pos = torch.rand(100, 3)
    batch = torch.zeros(pos.size(0), dtype=torch.long)

    with raises(AssertionError):
        dist(pos, batch)

    # example data where the second molecule should produce an error
    # due to exceeding the maximum number of neighbors
    pos = torch.rand(100, 3)
    batch = torch.tensor([0] * 20 + [1] * 80, dtype=torch.long)

    with raises(AssertionError):
        dist(pos, batch)


def test_gated_eq_gradients():
    model = create_model(
        load_example_args(
            "equivariant-transformer", prior_model=None, cutoff_upper=5, derivative=True
        )
    )

    # generate example where one atom is outside the cutoff radius of all others
    z = torch.tensor([1, 1, 8])
    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=torch.float)

    _, forces = model(z, pos)

    # compute gradients of forces with respect to the model's emebdding weights
    deriv = grad(forces.sum(), model.representation_model.embedding.weight)[0]
    assert (
        not deriv.isnan().any()
    ), "Encountered NaN gradients while backpropagating the force loss"
