from turtle import pos
import numpy as np
import pytest
import torch as pt
from torchmdnet.neighbors import get_neighbor_list

@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100, 1000, 10000])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_neighbors(num_atoms, device):

    positions = pt.randn((num_atoms, 3), device=device)
    device = positions.device

    ref_rows, ref_columns = np.tril_indices(num_atoms, -1)
    ref_positions = positions.cpu().numpy()
    ref_distances = np.linalg.norm(ref_positions[ref_rows] - ref_positions[ref_columns], axis=1)

    rows, columns, distances = get_neighbor_list(positions)

    assert rows.device == device
    assert columns.device == device
    assert distances.device == device

    assert rows.dtype == pt.int32
    assert columns.dtype == pt.int32
    assert distances.dtype == pt.float32

    assert np.all(ref_rows == rows.cpu().numpy())
    assert np.all(ref_columns == columns.cpu().numpy())
    assert np.allclose(ref_distances, distances.cpu().numpy())