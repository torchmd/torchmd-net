import torch
import numpy as np

from ase.neighborlist import neighbor_list
from ase import Atoms

def ase_neighbor_list(data, rcut, self_interaction=False):
    assert data.n_atoms.shape[0] == 1, 'data should contain only one structure'

    frame = Atoms(positions=data.pos.numpy(), cell=data.cell.numpy(),
                     pbc=data.pbc.numpy(), numbers=data.z.numpy())

    idx_i, idx_j, idx_S = neighbor_list(
            "ijS", frame, cutoff=rcut, self_interaction=self_interaction
        )
    return torch.from_numpy(idx_i), torch.from_numpy(idx_j), torch.from_numpy(np.dot(idx_S, frame.cell))
