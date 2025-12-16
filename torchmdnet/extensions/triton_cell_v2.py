"""
Efficient cell list neighbor list implementation for Triton.

Key design:
- One program per cell (not per atom)
- Fixed-size padded cell buffers for CUDA graph compatibility
- O(n) complexity: each cell examines only 27 neighboring cells
"""

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

import torch
from torch import Tensor
from typing import Tuple

from torchmdnet.extensions.triton_neighbors import TritonNeighborAutograd
from torchmdnet.extensions.triton_cell import _get_cell_dimensions


@triton.jit
def _tl_round(x):
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


@triton.jit
def cell_neighbor_kernel(
    # Cell data structure (1D sorted approach)
    SortedIndices,  # [n_atoms] - original atom indices, sorted by cell
    CellStart,  # [num_cells] - start index in SortedIndices for each cell
    CellEnd,  # [num_cells] - end index (exclusive) in SortedIndices for each cell
    Positions,  # [n_atoms, 3] - positions (original order)
    Batch,  # [n_atoms] - batch indices (original order)
    # Box parameters
    BoxSizes,  # [3] - box dimensions
    CellDims,  # [3] - number of cells in each dimension (int32)
    # Output
    OutPairs,  # [2, max_pairs]
    OutDeltas,  # [max_pairs, 3]
    OutDists,  # [max_pairs]
    GlobalCounter,  # [1]
    # Scalar parameters
    max_pairs,
    cutoff_lower_sq,
    cutoff_upper_sq,
    # Flags
    use_periodic: tl.constexpr,
    loop: tl.constexpr,
    include_transpose: tl.constexpr,
):
    """
    Each program processes one cell (the "home" cell).
    Uses 1D sorted array with cell_start/cell_end pointers.
    Iterates only over actual atoms in each cell using while loops.

    To avoid double-counting:
    - For half list (include_transpose=False): only emit pairs where home_atom > neighbor_atom
    - For full list (include_transpose=True): emit both directions
    """
    home_cell_id = tl.program_id(0)

    # Load box and cell dimensions
    box_x = tl.load(BoxSizes + 0)
    box_y = tl.load(BoxSizes + 1)
    box_z = tl.load(BoxSizes + 2)

    num_cells_x = tl.load(CellDims + 0)
    num_cells_y = tl.load(CellDims + 1)
    num_cells_z = tl.load(CellDims + 2)

    # Decompose home cell ID into 3D coordinates
    cells_yz = num_cells_y * num_cells_z
    home_cx = home_cell_id // cells_yz
    home_cy = (home_cell_id % cells_yz) // num_cells_z
    home_cz = home_cell_id % num_cells_z

    # Load home cell boundaries
    home_start = tl.load(CellStart + home_cell_id)
    home_end = tl.load(CellEnd + home_cell_id)

    # Loop over 27 neighbor cells
    for neighbor_offset in tl.range(0, 27):
        # Decompose neighbor_offset into di, dj, dk (each in {-1, 0, 1})
        di = (neighbor_offset % 3) - 1
        dj = ((neighbor_offset // 3) % 3) - 1
        dk = (neighbor_offset // 9) - 1

        # Compute neighbor cell coordinates
        ni = home_cx + di
        nj = home_cy + dj
        nk = home_cz + dk

        # Handle boundary conditions
        if use_periodic:
            ni = (ni + num_cells_x) % num_cells_x
            nj = (nj + num_cells_y) % num_cells_y
            nk = (nk + num_cells_z) % num_cells_z
            cell_valid = True
        else:
            cell_valid = (
                (ni >= 0)
                & (ni < num_cells_x)
                & (nj >= 0)
                & (nj < num_cells_y)
                & (nk >= 0)
                & (nk < num_cells_z)
            )

        neighbor_cell_id = ni * cells_yz + nj * num_cells_z + nk

        # Load neighbor cell boundaries
        neighbor_start = tl.load(CellStart + neighbor_cell_id)
        neighbor_end = tl.load(CellEnd + neighbor_cell_id)

        # If cell is invalid (non-periodic boundary), make it empty
        neighbor_start = tl.where(cell_valid, neighbor_start, 0)
        neighbor_end = tl.where(cell_valid, neighbor_end, 0)

        # Iterate over home atoms using while loop (only actual atoms)
        home_i = home_start
        while home_i < home_end:
            # Load home atom data
            home_atom = tl.load(SortedIndices + home_i)
            home_x = tl.load(Positions + home_atom * 3 + 0)
            home_y = tl.load(Positions + home_atom * 3 + 1)
            home_z = tl.load(Positions + home_atom * 3 + 2)
            home_batch = tl.load(Batch + home_atom)

            # Iterate over neighbor atoms using while loop (only actual atoms)
            neighbor_i = neighbor_start
            while neighbor_i < neighbor_end:
                # Load neighbor atom data
                neighbor_atom = tl.load(SortedIndices + neighbor_i)
                neighbor_x = tl.load(Positions + neighbor_atom * 3 + 0)
                neighbor_y = tl.load(Positions + neighbor_atom * 3 + 1)
                neighbor_z = tl.load(Positions + neighbor_atom * 3 + 2)
                neighbor_batch = tl.load(Batch + neighbor_atom)

                # Compute distance
                dx = home_x - neighbor_x
                dy = home_y - neighbor_y
                dz = home_z - neighbor_z

                # Apply PBC
                if use_periodic:
                    dx = dx - box_x * _tl_round(dx / box_x)
                    dy = dy - box_y * _tl_round(dy / box_y)
                    dz = dz - box_z * _tl_round(dz / box_z)

                dist_sq = dx * dx + dy * dy + dz * dz

                # Check validity
                cond_dist = (dist_sq < cutoff_upper_sq) & (dist_sq >= cutoff_lower_sq)
                cond_batch = home_batch == neighbor_batch

                # Index ordering to avoid double-counting
                if include_transpose:
                    if loop:
                        cond_idx = True
                    else:
                        cond_idx = home_atom != neighbor_atom
                else:
                    if loop:
                        cond_idx = home_atom >= neighbor_atom
                    else:
                        cond_idx = home_atom > neighbor_atom

                is_valid = cond_dist & cond_batch & cond_idx

                if is_valid:
                    # Atomically reserve one slot
                    current_offset = tl.atomic_add(GlobalCounter, 1)

                    if current_offset < max_pairs:
                        dist = tl.sqrt(dist_sq)
                        # Store pair
                        tl.store(OutPairs + 0 * max_pairs + current_offset, home_atom)
                        tl.store(
                            OutPairs + 1 * max_pairs + current_offset, neighbor_atom
                        )
                        tl.store(OutDeltas + current_offset * 3 + 0, dx)
                        tl.store(OutDeltas + current_offset * 3 + 1, dy)
                        tl.store(OutDeltas + current_offset * 3 + 2, dz)
                        tl.store(OutDists + current_offset, dist)

                neighbor_i += 1
            home_i += 1


def build_cell_list(
    positions: Tensor,
    box_sizes: Tensor,  # [3] diagonal elements
    use_periodic: bool,
    cell_dims: Tensor,  # [3] number of cells in each dimension
    num_cells: int,  # total number of cells (fixed for CUDA graphs)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build the cell list data structure using 1D sorted arrays.

    Args:
        positions: [N, 3] atom positions
        box_sizes: [3] box diagonal elements
        use_periodic: whether to use periodic boundary conditions
        cell_dims: [3] number of cells in each dimension (pre-computed)
        num_cells: total number of cells (pre-computed, fixed for CUDA graphs)

    Returns:
        sorted_indices: [n_atoms] - original atom indices, sorted by cell
        cell_start: [num_cells] - start index in sorted_indices for each cell
        cell_end: [num_cells] - end index (exclusive) in sorted_indices for each cell
    """
    device = positions.device
    n_atoms = positions.size(0)

    # Compute cell index for each atom
    if use_periodic:
        # Wrap to [0, box)
        inv_box = 1.0 / box_sizes
        wrapped = positions - torch.floor(positions * inv_box) * box_sizes
    else:
        # Shift by half box (like CUDA implementation)
        wrapped = positions + 0.5 * box_sizes

    # Cell coordinates
    cell_size = box_sizes / cell_dims.float()
    cell_coords = (wrapped / cell_size).long()
    cell_coords = torch.clamp(
        cell_coords, min=torch.zeros(3, device=device), max=cell_dims - 1
    )

    # Flat cell index
    cell_idx = (
        cell_coords[:, 0] * (cell_dims[1] * cell_dims[2])
        + cell_coords[:, 1] * cell_dims[2]
        + cell_coords[:, 2]
    ).long()

    # Sort atoms by cell index
    sorted_cell_idx, sort_order = torch.sort(cell_idx)
    sorted_indices = sort_order.int()  # Original atom indices, now sorted by cell

    # Count atoms per cell
    cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_counts.scatter_add_(
        0, cell_idx, torch.ones(n_atoms, dtype=torch.int32, device=device)
    )

    # Compute cell_start and cell_end using cumsum
    cell_end = torch.cumsum(cell_counts, dim=0).int()
    cell_start = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_start[1:] = cell_end[:-1]

    return sorted_indices, cell_start, cell_end


class TritonCellNeighborV2(TritonNeighborAutograd):
    @staticmethod
    def forward(
        ctx,
        positions: Tensor,
        batch: Tensor,
        box_vectors: Tensor,
        use_periodic: bool,
        cutoff_lower: float,
        cutoff_upper: float,
        max_num_pairs: int,
        loop: bool,
        include_transpose: bool,
        num_cells: int,
    ):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not available")

        device = positions.device
        dtype = positions.dtype
        n_atoms = positions.size(0)

        # Validate inputs
        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError('Expected "positions" to have shape (N, 3)')
        if batch.dim() != 1 or batch.size(0) != n_atoms:
            raise ValueError('Expected "batch" to have shape (N,)')

        # Extract box diagonal
        box_vectors = box_vectors.contiguous()
        if box_vectors.dim() == 3:
            box_diag = box_vectors[0]
        else:
            box_diag = box_vectors
        box_sizes = torch.stack(
            [box_diag[0, 0], box_diag[1, 1], box_diag[2, 2]]
        ).contiguous()

        # Compute cell dimensions using shared utility (stays on GPU)
        cell_dims = _get_cell_dimensions(
            box_sizes[0], box_sizes[1], box_sizes[2], cutoff_upper
        )

        # Build cell list (1D sorted approach)
        sorted_indices, cell_start, cell_end = build_cell_list(
            positions, box_sizes, use_periodic, cell_dims, num_cells
        )

        # Allocate outputs
        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        # Launch kernel: one program per cell
        grid = (num_cells,)
        cell_neighbor_kernel[grid](
            sorted_indices,
            cell_start,
            cell_end,
            positions,
            batch,
            box_sizes,
            cell_dims,
            neighbors,
            deltas,
            distances,
            counter,
            max_num_pairs,
            cutoff_lower**2,
            cutoff_upper**2,
            use_periodic=use_periodic,
            loop=loop,
            include_transpose=include_transpose,
        )
        num_pairs = counter.to(torch.long)

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        # Call parent backward (returns 9 values) and add None for num_cells
        parent_grads = TritonNeighborAutograd.backward(
            ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs
        )
        return (*parent_grads, None)
