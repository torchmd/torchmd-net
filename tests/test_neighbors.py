import pytest
import torch as pt
from math import sqrt
from moleculekit.molecule import Molecule
from torchmdnet.neighbors import get_neighbor_list

@pytest.mark.parametrize('molecule_file', ['alanine_dipeptide.pdb', 'testosterone.pdb', 'chignolin.pdb', 'dhfr.pdb'])
@pytest.mark.parametrize('radius', [8, 10, 12])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if pt.cuda.is_available() else []))
def test_neighbors(molecule_file, radius, device):
    radius_squared = radius * radius

    molecule_file = 'benchmarks/systems/' + molecule_file

    molecule = Molecule(molecule_file)
    positions = pt.tensor(molecule.coords[:,:,0], dtype=pt.float32, device=device)

    def get_distance_squared(p1, p2):
        diff_x = p2[0] - p1[0]
        diff_y = p2[1] - p1[1]
        diff_z = p2[2] - p1[2]

        return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z

    def naive_get_neighbor_list(positions):
        n_atoms = len(positions)
        rows = []
        columns = []
        distances = []

        for i in range(0, n_atoms):
            for j in range(i + 1, n_atoms):
                distance_squared = get_distance_squared(positions[i], positions[j])
                if distance_squared < radius_squared:
                    rows.append(i)
                    columns.append(j)
                    distances.append(sqrt(distance_squared))

        return rows, columns, distances

    naive_rows, naive_columns, _ = naive_get_neighbor_list(positions.cpu())
    rows, columns, _ = get_neighbor_list(positions, radius, 500)

    pairs = list((i, j) if i < j else (j, i) for i, j in zip(rows.tolist(), columns.tolist()))
    pairs.sort()
    naive_pairs = list(zip(naive_rows, naive_columns))

    assert len(pairs) == len(naive_pairs)
    assert all(i == j for i, j in zip(pairs, naive_pairs))
