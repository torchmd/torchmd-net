
from torch_geometric.data import Data
import torch


def ase2data(frame, energy_tag=None, force_tag=None):
    z = torch.from_numpy(frame.get_atomic_numbers())
    pos = torch.from_numpy(frame.get_positions())
    pbc  = torch.from_numpy(frame.get_pbc())
    cell = torch.tensor(frame.get_cell().tolist(), dtype=torch.float64)
    n_atoms = torch.tensor([len(frame)])
    data = Data(z=z, pos=pos, pbc=pbc, cell=cell, n_atoms=n_atoms)

    if energy_tag is not None:
        E = torch.tensor(frame.info[energy_tag])
        data.energy = E
    if force_tag is not None:
        forces = torch.from_numpy(frame.arrays[force_tag])
        data.forces = forces

    return data

def wrap_positions(data, eps=1e-7):
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.

    Parameters:

    data:
        torch_geometric.Data
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    """
    center=torch.tensor((0.5, 0.5, 0.5)).view(1, 3)
    assert data.n_atoms.shape[0] == 1, f"There should be only one structure, found: {data.n_atoms.shape[0]}"

    pbc = data.pbc.view(1,3)
    shift = center - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[torch.logical_not(pbc)] = 0.0

    # assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    cell = data.cell
    positions = data.pos

    fractional = torch.linalg.solve(cell.t(),
                                 positions.t()).t() - shift

    for i, periodic in enumerate(pbc.view(-1)):
        if periodic:
            fractional[:, i] = torch.remainder(fractional[:, i], 1.0)
            fractional[:, i] += shift[0, i]

    data.pos = torch.matmul(fractional, cell)

