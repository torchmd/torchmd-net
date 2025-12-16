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
    # Cell data structure
    CellAtoms,  # [num_cells, MAX_ATOMS_PER_CELL] - atom indices in each cell
    CellCounts,  # [num_cells] - number of atoms in each cell
    Positions,  # [n_atoms, 3] - positions (original order)
    Batch,  # [n_atoms] - batch indices
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
    # Block sizes
    MAX_ATOMS_PER_CELL: tl.constexpr,
):
    """
    Each program processes one cell (the "home" cell).
    It finds all pairs where the first atom is in the home cell
    and the second atom is in one of the 27 neighboring cells (including home).

    To avoid double-counting, we use the convention:
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

    # Load home cell data
    home_count = tl.load(CellCounts + home_cell_id)
    home_idx = tl.arange(0, MAX_ATOMS_PER_CELL)
    home_mask = home_idx < home_count

    # Load atom indices for home cell
    home_atoms = tl.load(
        CellAtoms + home_cell_id * MAX_ATOMS_PER_CELL + home_idx,
        mask=home_mask,
        other=0,
    )

    # Load positions for home atoms (gather)
    home_x = tl.load(Positions + home_atoms * 3 + 0, mask=home_mask, other=0.0)
    home_y = tl.load(Positions + home_atoms * 3 + 1, mask=home_mask, other=0.0)
    home_z = tl.load(Positions + home_atoms * 3 + 2, mask=home_mask, other=0.0)
    home_batch = tl.load(Batch + home_atoms, mask=home_mask, other=-1)

    # Loop over 27 neighbor cells
    # Use tl.range() for runtime loop (not unrolled at compile time)
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
            # Wrap around with PBC
            ni = (ni + num_cells_x) % num_cells_x
            nj = (nj + num_cells_y) % num_cells_y
            nk = (nk + num_cells_z) % num_cells_z
            cell_valid = True
        else:
            # Check bounds for non-periodic
            cell_valid = (
                (ni >= 0)
                & (ni < num_cells_x)
                & (nj >= 0)
                & (nj < num_cells_y)
                & (nk >= 0)
                & (nk < num_cells_z)
            )

        # Skip invalid cells (non-periodic boundary)
        # Note: We can't early-continue in Triton, so we use masking
        neighbor_cell_id = ni * cells_yz + nj * num_cells_z + nk

        # Load neighbor cell data
        neighbor_count = tl.load(CellCounts + neighbor_cell_id)
        # Apply cell_valid to count (if cell invalid, treat as empty)
        neighbor_count = tl.where(cell_valid, neighbor_count, 0)

        neighbor_idx = tl.arange(0, MAX_ATOMS_PER_CELL)
        neighbor_mask = neighbor_idx < neighbor_count

        # Load atom indices for neighbor cell
        neighbor_atoms = tl.load(
            CellAtoms + neighbor_cell_id * MAX_ATOMS_PER_CELL + neighbor_idx,
            mask=neighbor_mask,
            other=0,
        )

        # Load positions for neighbor atoms
        neighbor_x = tl.load(
            Positions + neighbor_atoms * 3 + 0, mask=neighbor_mask, other=0.0
        )
        neighbor_y = tl.load(
            Positions + neighbor_atoms * 3 + 1, mask=neighbor_mask, other=0.0
        )
        neighbor_z = tl.load(
            Positions + neighbor_atoms * 3 + 2, mask=neighbor_mask, other=0.0
        )
        neighbor_batch = tl.load(Batch + neighbor_atoms, mask=neighbor_mask, other=-2)

        # Compute pairwise distances: [MAX_ATOMS_PER_CELL, MAX_ATOMS_PER_CELL]
        # home atoms are rows, neighbor atoms are columns
        dx = home_x[:, None] - neighbor_x[None, :]
        dy = home_y[:, None] - neighbor_y[None, :]
        dz = home_z[:, None] - neighbor_z[None, :]

        # Apply PBC to distance vectors
        if use_periodic:
            dx = dx - box_x * _tl_round(dx / box_x)
            dy = dy - box_y * _tl_round(dy / box_y)
            dz = dz - box_z * _tl_round(dz / box_z)

        dist_sq = dx * dx + dy * dy + dz * dz

        # Build validity mask
        # 1. Distance within cutoff
        cond_dist = (dist_sq < cutoff_upper_sq) & (dist_sq >= cutoff_lower_sq)

        # 2. Same batch
        cond_batch = home_batch[:, None] == neighbor_batch[None, :]

        # 3. Index ordering to avoid double-counting
        home_atoms_bc = home_atoms[:, None]
        neighbor_atoms_bc = neighbor_atoms[None, :]

        if include_transpose:
            # Full list: emit both (i,j) and (j,i), but only from one cell
            # We emit when home_atom != neighbor_atom (self-loops handled by loop flag)
            if loop:
                cond_idx = True  # Include self-loops
            else:
                cond_idx = home_atoms_bc != neighbor_atoms_bc
        else:
            # Half list: only emit (i,j) where i > j
            # This ensures each pair is emitted exactly once
            if loop:
                cond_idx = home_atoms_bc >= neighbor_atoms_bc
            else:
                cond_idx = home_atoms_bc > neighbor_atoms_bc

        # 4. Both atoms must be valid (within cell counts)
        cond_valid = home_mask[:, None] & neighbor_mask[None, :]

        # Combined validity
        valid_mask = cond_dist & cond_batch & cond_idx & cond_valid

        # Count and store valid pairs
        num_found = tl.sum(valid_mask.to(tl.int32))

        if num_found > 0:
            # Atomically reserve space in output
            current_offset = tl.atomic_add(GlobalCounter, num_found)

            if current_offset + num_found <= max_pairs:
                # Compute storage indices using cumsum
                flat_mask = tl.ravel(valid_mask)
                csum = tl.cumsum(flat_mask.to(tl.int32), axis=0)
                store_idx = current_offset + csum - 1

                # Prepare flattened data
                flat_home = tl.ravel(
                    tl.broadcast_to(
                        home_atoms[:, None],
                        (MAX_ATOMS_PER_CELL, MAX_ATOMS_PER_CELL),
                    )
                )
                flat_neighbor = tl.ravel(
                    tl.broadcast_to(
                        neighbor_atoms[None, :],
                        (MAX_ATOMS_PER_CELL, MAX_ATOMS_PER_CELL),
                    )
                )
                flat_dx = tl.ravel(dx)
                flat_dy = tl.ravel(dy)
                flat_dz = tl.ravel(dz)
                flat_dist = tl.sqrt(tl.ravel(dist_sq))

                # Store pairs
                tl.store(
                    OutPairs + 0 * max_pairs + store_idx,
                    flat_home,
                    mask=flat_mask,
                )
                tl.store(
                    OutPairs + 1 * max_pairs + store_idx,
                    flat_neighbor,
                    mask=flat_mask,
                )

                # Store deltas (interleaved x,y,z)
                tl.store(OutDeltas + store_idx * 3 + 0, flat_dx, mask=flat_mask)
                tl.store(OutDeltas + store_idx * 3 + 1, flat_dy, mask=flat_mask)
                tl.store(OutDeltas + store_idx * 3 + 2, flat_dz, mask=flat_mask)

                # Store distances
                tl.store(OutDists + store_idx, flat_dist, mask=flat_mask)


def build_cell_list(
    positions: Tensor,
    box_sizes: Tensor,  # [3] diagonal elements
    use_periodic: bool,
    cell_dims: Tensor,  # [3] number of cells in each dimension
    num_cells: int,  # total number of cells (fixed for CUDA graphs)
    max_atoms_per_cell: int,
) -> Tuple[Tensor, Tensor]:
    """
    Build the cell list data structure.

    Args:
        positions: [N, 3] atom positions
        batch: [N] batch indices
        box_sizes: [3] box diagonal elements
        cutoff: cutoff distance
        use_periodic: whether to use periodic boundary conditions
        cell_dims: [3] number of cells in each dimension (pre-computed)
        num_cells: total number of cells (pre-computed, fixed for CUDA graphs)
        max_atoms_per_cell: maximum atoms per cell

    Returns:
        cell_atoms: [num_cells, max_atoms_per_cell] - atom indices per cell
        cell_counts: [num_cells] - number of atoms in each cell
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

    # Count atoms per cell
    cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_counts.scatter_add_(
        0, cell_idx, torch.ones(n_atoms, dtype=torch.int32, device=device)
    )

    # Check for overflow
    if not torch.cuda.is_current_stream_capturing():
        max_count = cell_counts.max().item()
        if max_count > max_atoms_per_cell:
            raise RuntimeError(
                f"Cell overflow: {max_count} atoms in one cell, but max_atoms_per_cell={max_atoms_per_cell}. "
                f"Increase max_atoms_per_cell or use a larger cutoff."
            )

    # Build cell_atoms array using sorting approach
    # Sort atoms by cell index
    sorted_cell_idx, sort_order = torch.sort(cell_idx)

    # Compute starting position for each cell
    cell_starts = torch.zeros(num_cells + 1, dtype=torch.long, device=device)
    cell_starts[1:] = torch.cumsum(cell_counts.long(), dim=0)

    # Initialize cell_atoms with -1 (invalid)
    cell_atoms = torch.full(
        (num_cells, max_atoms_per_cell), -1, dtype=torch.int32, device=device
    )

    # Fill cell_atoms using a vectorized approach
    # Create position within cell for each atom
    atom_positions = torch.arange(n_atoms, device=device)
    # Position within cell = atom_position - cell_start[cell_of_atom]
    position_in_cell = atom_positions - cell_starts[sorted_cell_idx]

    # Scatter atom indices into cell_atoms
    # cell_atoms[sorted_cell_idx, position_in_cell] = sort_order
    flat_idx = sorted_cell_idx * max_atoms_per_cell + position_in_cell
    cell_atoms.view(-1).scatter_(0, flat_idx, sort_order.int())

    return cell_atoms, cell_counts


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
        max_atoms_per_cell: int = 128,
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

        # Build cell list
        cell_atoms, cell_counts = build_cell_list(
            positions, box_sizes, use_periodic, cell_dims, num_cells, max_atoms_per_cell
        )

        # Allocate outputs
        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        # Launch kernel: one program per cell
        grid = (num_cells,)
        cell_neighbor_kernel[grid](
            cell_atoms,
            cell_counts,
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
            MAX_ATOMS_PER_CELL=max_atoms_per_cell,
        )
        num_pairs = counter.to(torch.long)

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        # Call parent backward (returns 9 values) and add None for num_cells and max_atoms_per_cell
        parent_grads = TritonNeighborAutograd.backward(
            ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs
        )
        return (*parent_grads, None, None)


# def triton_cell_neighbor_v2(
#     positions: Tensor,
#     batch: Tensor,
#     box_vectors: Tensor,
#     use_periodic: bool,
#     cutoff_lower: float,
#     cutoff_upper: float,
#     max_num_pairs: int,
#     loop: bool,
#     include_transpose: bool,
#     num_cells: int,
#     max_atoms_per_cell: int = 64,
# ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#     """
#     Efficient cell list neighbor list using Triton.

#     This implementation achieves O(n) complexity by:
#     1. Spawning one program per cell
#     2. Using fixed-size cell buffers (MAX_ATOMS_PER_CELL)
#     3. Each program only examines the 27 neighboring cells

#     Args:
#         positions: [N, 3] atom positions
#         batch: [N] batch indices
#         box_vectors: [3, 3] or [n_batch, 3, 3] box vectors (must be diagonal)
#         use_periodic: whether to use periodic boundary conditions
#         cutoff_lower: lower cutoff distance
#         cutoff_upper: upper cutoff distance
#         max_num_pairs: maximum number of pairs to return
#         loop: whether to include self-interactions (i == j)
#         include_transpose: whether to include both (i,j) and (j,i)
#         num_cells: total number of cells (fixed for CUDA graph compatibility)
#         max_atoms_per_cell: maximum atoms per cell (default 64)

#     Returns:
#         neighbors: [2, num_pairs] pair indices
#         deltas: [num_pairs, 3] displacement vectors
#         distances: [num_pairs] distances
#         num_pairs: [1] number of pairs found
#     """
#     return TritonCellNeighborV2.apply(
#         positions,
#         batch,
#         box_vectors,
#         use_periodic,
#         cutoff_lower,
#         cutoff_upper,
#         max_num_pairs,
#         loop,
#         include_transpose,
#         num_cells,
#         max_atoms_per_cell,
#     )
