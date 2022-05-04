from os.path import join, exists
from pytest import mark, raises
import torch
from torchmdnet.models.utils import Distance


@mark.parametrize("cutoff_lower", [0, 2])
@mark.parametrize("cutoff_upper", [5, 10])
@mark.parametrize("return_vecs", [False, True])
@mark.parametrize("loop", [False, True])
def test_make_splits_ratios(cutoff_lower, cutoff_upper, return_vecs, loop):
    dist = Distance(cutoff_lower, cutoff_upper, return_vecs, loop)
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

    if cutoff_lower > 1:
        assert edge_index.size(1) == loop_extra, "Expected only self loops"
    else:
        assert edge_index.size(1) == (
            len(batch) * (len(batch) - 1) + loop_extra
        ), "Expected all neighbors to match"
