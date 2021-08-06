import torch

from torch_cluster import radius, radius_graph

from typing import Tuple


def torch_neighbor_list(data, rcut, self_interaction=True, num_workers=1, max_num_neighbors=1000):
    if 'pbc' in data:
        pbc = data.pbc
    else:
        pbc = torch.zeros(3, dtype=bool)

    if torch.any(pbc):
        if 'cell' not in data:
            raise ValueError(f'Periodic systems need to have a unit cell defined')
        idx_i, idx_j, cell_shifts, self_interaction_mask = torch_neighbor_list_pbc(data, rcut, self_interaction=self_interaction, num_workers=num_workers, max_num_neighbors=max_num_neighbors)
    else:
        idx_i, idx_j, self_interaction_mask = torch_neighbor_list_no_pbc(data, rcut, self_interaction=self_interaction,
                                               num_workers=num_workers, max_num_neighbors=max_num_neighbors)
        cell_shifts = torch.zeros((idx_i.shape[0], 3), dtype=data.pos.dtype, device=data.pos.device)

    return idx_i, idx_j, cell_shifts, self_interaction_mask

@torch.jit.script
def compute_images(positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, batch: torch.Tensor, n_atoms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = cell.view((-1, 3, 3)).to(torch.float64)
    pbc = pbc.view((-1, 3))
    reciprocal_cell = torch.linalg.inv(cell).transpose(2, 1)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats_ = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    images, batch_images, shifts_expanded, shifts_idx_ = [], [], [], []
    for i_structure, num_repeats in enumerate(num_repeats_):
        r1 = torch.arange(-num_repeats[0], num_repeats[0] + 1, device=cell.device, dtype=torch.long)
        r2 = torch.arange(-num_repeats[1], num_repeats[1] + 1, device=cell.device, dtype=torch.long)
        r3 = torch.arange(-num_repeats[2], num_repeats[2] + 1, device=cell.device, dtype=torch.long)
        shifts_idx = torch.cartesian_prod(r1, r2, r3)
        shifts = torch.matmul(shifts_idx.to(cell.dtype), cell[i_structure])
        pos = positions[batch == i_structure]
        shift_expanded = shifts.repeat(1, n_atoms[i_structure]).view((-1, 3))
        pos_expanded = pos.repeat(shifts.shape[0], 1)
        images.append(pos_expanded + shift_expanded)

        batch_images.append(i_structure*torch.ones(images[-1].shape[0], dtype=torch.int64, device=cell.device))
        shifts_expanded.append(shift_expanded)
        shifts_idx_.append(shifts_idx.repeat(1, n_atoms[i_structure]).view((-1, 3)))
    return (torch.cat(images, dim=0), torch.cat(batch_images, dim=0),
                torch.cat(shifts_expanded, dim=0), torch.cat(shifts_idx_, dim=0))

@torch.jit.script
def strides_of(v: torch.Tensor) -> torch.Tensor:
    strides = torch.zeros(v.shape[0]+1, dtype=torch.int64, device=v.device)
    strides[1:] = v
    strides = torch.cumsum(strides, dim=0)
    return strides

def torch_neighbor_list_no_pbc(data, rcut, self_interaction=True, num_workers=1, max_num_neighbors=1000):
    # assert data.n_atoms.shape[0] == 1, 'data should contain only one structure'

    edge_index = radius_graph(data.pos, rcut, batch=data.batch, max_num_neighbors = max_num_neighbors,
                        num_workers=num_workers, flow='target_to_source', loop=self_interaction)
    self_interaction_mask = edge_index[0] != edge_index[1]
    return edge_index[0], edge_index[1], self_interaction_mask



@torch.jit.script
def get_j_idx(edge_index: torch.Tensor, batch_images:torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    # get neighbor index reffering to the list of original positions
    n_neighbors = torch.bincount(edge_index[0])
    strides = strides_of(n_atoms)
    n_reapeats = torch.zeros_like(n_atoms)
    for i_structure,(st,nd) in enumerate(zip(strides[:-1], strides[1:])):
        n_reapeats[i_structure] = torch.sum(n_neighbors[st:nd])
    n_atoms = torch.repeat_interleave(n_atoms, n_reapeats, dim=0)

    batch_i = torch.repeat_interleave(strides[:-1], n_reapeats, dim=0)

    n_images = torch.bincount(batch_images)
    strides_images = strides_of(n_images[:-1])
    images_shift = torch.repeat_interleave(strides_images, n_reapeats, dim=0)

    j_idx = torch.remainder(edge_index[1]-images_shift, n_atoms) + batch_i
    return j_idx

def torch_neighbor_list_pbc(data, rcut, self_interaction=True, num_workers=1, max_num_neighbors=1000):
    images, batch_images, shifts_expanded, shifts_idx = compute_images(data.pos, data.cell, data.pbc, rcut, data.batch, data.n_atoms)
    edge_index = radius(x=images, y=data.pos, r=rcut, batch_x=batch_images, batch_y=data.batch,
                        max_num_neighbors = max_num_neighbors,
                        num_workers = num_workers)

    j_idx = get_j_idx(edge_index,batch_images, data.n_atoms)

    # find self interactions
    is_central_cell = (shifts_idx[edge_index[1]] == 0).all(dim=1)
    mask = torch.cat([is_central_cell.view(-1,1), (edge_index[0] == j_idx).view(-1,1)], dim=1)
    self_interaction_mask = torch.logical_not(torch.all(mask,dim=1))

    if self_interaction:
        idx_i, idx_j = edge_index[0], j_idx
        cell_shifts = shifts_expanded[edge_index[1]]
    else:
        # remove self interaction
        idx_i, idx_j = edge_index[0][self_interaction_mask], j_idx[self_interaction_mask]
        cell_shifts = shifts_expanded[edge_index[1][self_interaction_mask]]

    return idx_i, idx_j, cell_shifts, self_interaction_mask


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



