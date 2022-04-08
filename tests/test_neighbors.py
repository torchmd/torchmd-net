from turtle import pos
import numpy as np
import pytest
import torch as pt
from torchmdnet.neighbors import get_neighbor_pairs

@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100, 1000, 10000])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_neighbors(num_atoms, device):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    positions = pt.randn((num_atoms, 3), device=device)
    device = positions.device

    ref_neighbors = np.tril_indices(num_atoms, -1)
    ref_positions = positions.cpu().numpy()
    ref_distances = np.linalg.norm(ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]], axis=1)

    neighbors, distances = get_neighbor_pairs(positions)

    assert neighbors.device == device
    assert distances.device == device

    assert neighbors.dtype == pt.int32
    assert distances.dtype == pt.float32

    assert np.all(ref_neighbors == neighbors.cpu().numpy())
    assert np.allclose(ref_distances, distances.cpu().numpy())