# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple
from torchmdnet.extensions.triton_neighbors import TritonNeighborAutograd
from torch.library import triton_op, wrap_triton


def _get_cell_dimensions(
    box_x: torch.float32,
    box_y: torch.float32,
    box_z: torch.float32,
    cutoff_upper: torch.float32,
) -> int:
    nx = torch.floor(box_x / cutoff_upper).clamp(min=3).long()
    ny = torch.floor(box_y / cutoff_upper).clamp(min=3).long()
    nz = torch.floor(box_z / cutoff_upper).clamp(min=3).long()
    return torch.stack([nx, ny, nz])


@triton.jit
def _tl_round(x):
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


@triton.jit
def cell_neighbor_kernel(
    # Cell data structure (1D sorted approach)
    SortedIndices,  # [n_atoms] - original atom indices, sorted by cell
    SortedPositions,  # [n_atoms, 3] - positions sorted by cell (for coalesced access)
    SortedBatch,  # [n_atoms] - batch indices sorted by cell
    CellStart,  # [num_cells] - start index in sorted arrays for each cell
    CellEnd,  # [num_cells] - end index (exclusive) in sorted arrays for each cell
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
    # Batch size for vectorized processing
    BATCH_SIZE: tl.constexpr,  # e.g., 32 -> processes 32×32=1024 pairs per iteration
):
    """
    Each program processes one cell (the "home" cell).
    Uses 1D sorted array with cell_start/cell_end pointers.

    Vectorized batched processing:
    - Loads BATCH_SIZE atoms at a time for both home and neighbor
    - Computes BATCH_SIZE × BATCH_SIZE distance matrix per iteration
    - Uses while loops to iterate only over actual atoms
    - Minimal waste: only last partial batch may have masked elements

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

        # Batched iteration over home atoms
        home_batch_start = home_start
        while home_batch_start < home_end:
            # Load BATCH_SIZE home atoms
            home_offsets = tl.arange(0, BATCH_SIZE)
            home_global_idx = home_batch_start + home_offsets
            home_mask = home_global_idx < home_end

            # Load home atom original indices (for output pair indices)
            home_atoms = tl.load(
                SortedIndices + home_global_idx, mask=home_mask, other=0
            )

            # Load home atom positions (sequential access - coalesced!)
            home_x = tl.load(
                SortedPositions + home_global_idx * 3 + 0, mask=home_mask, other=0.0
            )
            home_y = tl.load(
                SortedPositions + home_global_idx * 3 + 1, mask=home_mask, other=0.0
            )
            home_z = tl.load(
                SortedPositions + home_global_idx * 3 + 2, mask=home_mask, other=0.0
            )
            home_batch = tl.load(
                SortedBatch + home_global_idx, mask=home_mask, other=-1
            )

            # Batched iteration over neighbor atoms
            neighbor_batch_start = neighbor_start
            while neighbor_batch_start < neighbor_end:
                # Load BATCH_SIZE neighbor atoms
                neighbor_offsets = tl.arange(0, BATCH_SIZE)
                neighbor_global_idx = neighbor_batch_start + neighbor_offsets
                neighbor_mask = neighbor_global_idx < neighbor_end

                # Load neighbor atom original indices (for output pair indices)
                neighbor_atoms = tl.load(
                    SortedIndices + neighbor_global_idx, mask=neighbor_mask, other=0
                )

                # Load neighbor atom positions (sequential access - coalesced!)
                neighbor_x = tl.load(
                    SortedPositions + neighbor_global_idx * 3 + 0,
                    mask=neighbor_mask,
                    other=0.0,
                )
                neighbor_y = tl.load(
                    SortedPositions + neighbor_global_idx * 3 + 1,
                    mask=neighbor_mask,
                    other=0.0,
                )
                neighbor_z = tl.load(
                    SortedPositions + neighbor_global_idx * 3 + 2,
                    mask=neighbor_mask,
                    other=0.0,
                )
                neighbor_batch_vals = tl.load(
                    SortedBatch + neighbor_global_idx, mask=neighbor_mask, other=-2
                )

                # Compute pairwise distances: [BATCH_SIZE, BATCH_SIZE]
                dx = home_x[:, None] - neighbor_x[None, :]
                dy = home_y[:, None] - neighbor_y[None, :]
                dz = home_z[:, None] - neighbor_z[None, :]

                # Apply PBC
                if use_periodic:
                    dx = dx - box_x * _tl_round(dx / box_x)
                    dy = dy - box_y * _tl_round(dy / box_y)
                    dz = dz - box_z * _tl_round(dz / box_z)

                dist_sq = dx * dx + dy * dy + dz * dz

                # Build validity mask
                # 1. Distance within cutoff
                cond_dist = (dist_sq < cutoff_upper_sq) & (dist_sq >= cutoff_lower_sq)

                # 2. Same batch
                cond_batch = home_batch[:, None] == neighbor_batch_vals[None, :]

                # 3. Index ordering to avoid double-counting
                home_atoms_bc = home_atoms[:, None]
                neighbor_atoms_bc = neighbor_atoms[None, :]

                if include_transpose:
                    if loop:
                        cond_idx = True
                    else:
                        cond_idx = home_atoms_bc != neighbor_atoms_bc
                else:
                    if loop:
                        cond_idx = home_atoms_bc >= neighbor_atoms_bc
                    else:
                        cond_idx = home_atoms_bc > neighbor_atoms_bc

                # 4. Both atoms must be valid (within actual cell bounds)
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
                                home_atoms[:, None], (BATCH_SIZE, BATCH_SIZE)
                            )
                        )
                        flat_neighbor = tl.ravel(
                            tl.broadcast_to(
                                neighbor_atoms[None, :], (BATCH_SIZE, BATCH_SIZE)
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

                        # Store deltas
                        tl.store(OutDeltas + store_idx * 3 + 0, flat_dx, mask=flat_mask)
                        tl.store(OutDeltas + store_idx * 3 + 1, flat_dy, mask=flat_mask)
                        tl.store(OutDeltas + store_idx * 3 + 2, flat_dz, mask=flat_mask)

                        # Store distances
                        tl.store(OutDists + store_idx, flat_dist, mask=flat_mask)

                neighbor_batch_start += BATCH_SIZE
            home_batch_start += BATCH_SIZE


def build_cell_list(
    positions: Tensor,
    batch: Tensor,
    box_sizes: Tensor,  # [3] diagonal elements
    use_periodic: bool,
    cell_dims: Tensor,  # [3] number of cells in each dimension
    num_cells: int,  # total number of cells (fixed for CUDA graphs)
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Build the cell list data structure using 1D sorted arrays.

    Args:
        positions: [N, 3] atom positions
        batch: [N] batch indices
        box_sizes: [3] box diagonal elements
        use_periodic: whether to use periodic boundary conditions
        cell_dims: [3] number of cells in each dimension (pre-computed)
        num_cells: total number of cells (pre-computed, fixed for CUDA graphs)

    Returns:
        sorted_indices: [n_atoms] - original atom indices, sorted by cell
        sorted_positions: [n_atoms, 3] - positions sorted by cell (for coalesced access)
        sorted_batch: [n_atoms] - batch indices sorted by cell
        cell_start: [num_cells] - start index in sorted arrays for each cell
        cell_end: [num_cells] - end index (exclusive) in sorted arrays for each cell
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

    # Create sorted positions and batch for coalesced memory access
    sorted_positions = positions.index_select(0, sort_order).contiguous()
    sorted_batch = batch.index_select(0, sort_order).contiguous()

    # Count atoms per cell
    cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_counts.scatter_add_(
        0, cell_idx, torch.ones(n_atoms, dtype=torch.int32, device=device)
    )

    # Compute cell_start and cell_end using cumsum
    cell_end = torch.cumsum(cell_counts, dim=0).int()
    cell_start = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_start[1:] = cell_end[:-1]

    return sorted_indices, sorted_positions, sorted_batch, cell_start, cell_end


@triton_op("torchmdnet::triton_neighbor_cell", mutates_args={})
def triton_neighbor_cell(
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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

    # Build cell list (1D sorted approach with sorted positions for coalesced access)
    sorted_indices, sorted_positions, sorted_batch, cell_start, cell_end = (
        build_cell_list(positions, batch, box_sizes, use_periodic, cell_dims, num_cells)
    )

    # Allocate outputs
    neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
    deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
    distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
    num_pairs = torch.zeros((1,), device=device, dtype=torch.int32)

    # Launch kernel: one program per cell
    # BATCH_SIZE: process atoms in batches for vectorized compute
    # 32 is a good balance: 32×32=1024 elements fits in registers, minimal waste on partial batches
    BATCH_SIZE = 32

    grid = (num_cells,)
    wrap_triton(cell_neighbor_kernel)[grid](
        sorted_indices,
        sorted_positions,
        sorted_batch,
        cell_start,
        cell_end,
        box_sizes,
        cell_dims,
        neighbors,
        deltas,
        distances,
        num_pairs,
        max_num_pairs,
        cutoff_lower**2,
        cutoff_upper**2,
        use_periodic=use_periodic,
        loop=loop,
        include_transpose=include_transpose,
        BATCH_SIZE=BATCH_SIZE,
    )
    return neighbors, deltas, distances, num_pairs


class TritonCellNeighborAutograd(TritonNeighborAutograd):
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
        neighbors, deltas, distances, num_pairs = triton_neighbor_cell(
            positions,
            batch,
            box_vectors,
            use_periodic,
            cutoff_lower,
            cutoff_upper,
            max_num_pairs,
            loop,
            include_transpose,
            num_cells,
        )

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = positions.size(0)
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        # Call parent backward (returns 9 values) and add None for num_cells
        parent_grads = TritonNeighborAutograd.backward(
            ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs
        )
        return (*parent_grads, None)
