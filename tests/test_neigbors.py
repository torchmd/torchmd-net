import pytest
from pytest import mark
from sklearn import neighbors
import torch as pt

from torchmdnet.models.utils import DistanceBruteForce, Distance

@mark.parametrize('num_atoms', [5, 7, 11, 13, 17])
@mark.parametrize('device', ['cpu', 'cuda'])
def test_neighbors(num_atoms, device):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    device = pt.device(device)

    # Generate random inputs
    pos = (10 * pt.rand(num_atoms, 3, dtype=pt.float32, device=device) - 5)

    simple = Distance(0.0, 100.0)
    brute_force = DistanceBruteForce()

    _, simple_distances, _ = simple(pos, None)
    _, brute_force_distances, _ = brute_force(pos, None)

    simple_distances = simple_distances.sort().values
    brute_force_distances = brute_force_distances.sort().values

    assert pt.allclose(simple_distances, brute_force_distances)