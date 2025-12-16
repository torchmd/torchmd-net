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
from torchmdnet.extensions.triton_neighbors import (
    _tl_round,
    TritonNeighborAutograd,
    _validate_box,
)


@triton.jit
def neighbor_list_kernel(
    # Pointers
    SortedCoords,
    SortedBatch,
    SortedToOrig,
    CellStart,
    CellEnd,
    OutPairs,
    OutDeltas,
    OutDists,
    GlobalCounter,
    BoxSizes,  # Pointer to [box_x, box_y, box_z]
    CellDims,  # Pointer to [n_cells_x, n_cells_y, n_cells_z]
    # Parameters
    n_atoms,
    max_pairs,
    cutoff_lower_sq,
    cutoff_upper_sq,
    # Flags
    use_periodic: tl.constexpr,  # Enable/Disable PBC
    loop: tl.constexpr,  # Enable/Disable Self-interactions
    include_transpose: tl.constexpr,  # True: i!=j (Full), False: i<j (Half)
    BLOCK_M: tl.constexpr,
    MAX_TILES: tl.constexpr,  # Maximum number of tiles for inner loop (n_atoms / BLOCK_M)
):
    pid = tl.program_id(0)
    idx_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = idx_m < n_atoms

    # Load box sizes (CUDA graph compatible - no CPU sync)
    box_x = tl.load(BoxSizes + 0)
    box_y = tl.load(BoxSizes + 1)
    box_z = tl.load(BoxSizes + 2)

    # Load cell dimensions (CUDA graph compatible - no CPU sync)
    n_cells_x = tl.load(CellDims + 0)
    n_cells_y = tl.load(CellDims + 1)
    n_cells_z = tl.load(CellDims + 2)

    # Load Query Data
    off_m = idx_m
    pos_mx = tl.load(SortedCoords + off_m * 3 + 0, mask=mask_m, other=0.0)
    pos_my = tl.load(SortedCoords + off_m * 3 + 1, mask=mask_m, other=0.0)
    pos_mz = tl.load(SortedCoords + off_m * 3 + 2, mask=mask_m, other=0.0)
    batch_m = tl.load(SortedBatch + off_m, mask=mask_m, other=-1)

    # Identify Home Cell - vectorized for all BLOCK_M atoms
    cell_sx = box_x / n_cells_x
    cell_sy = box_y / n_cells_y
    cell_sz = box_z / n_cells_z

    # Local coords for cell calculation (vectorized for all atoms in block)
    # Match CUDA implementation: for periodic, wrap to [0, box); for non-periodic, shift by 0.5*box
    if use_periodic:
        px = pos_mx - box_x * tl.math.floor(pos_mx / box_x)
        py = pos_my - box_y * tl.math.floor(pos_my / box_y)
        pz = pos_mz - box_z * tl.math.floor(pos_mz / box_z)
    else:
        # For non-periodic, shift by half box (like CUDA impl: (p + 0.5*box) / cutoff)
        px = pos_mx + 0.5 * box_x
        py = pos_my + 0.5 * box_y
        pz = pos_mz + 0.5 * box_z

    # Cell indices for each atom in the block (vectors of size BLOCK_M)
    c_i_vec = (px / cell_sx).to(tl.int32)
    c_j_vec = (py / cell_sy).to(tl.int32)
    c_k_vec = (pz / cell_sz).to(tl.int32)

    # Clamp to valid range
    c_i_vec = tl.maximum(tl.minimum(c_i_vec, n_cells_x - 1), 0)
    c_j_vec = tl.maximum(tl.minimum(c_j_vec, n_cells_y - 1), 0)
    c_k_vec = tl.maximum(tl.minimum(c_k_vec, n_cells_z - 1), 0)

    # Iterate over neighbor cell offsets
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):

                # Compute neighbor cell indices for all atoms in block (vectorized)
                n_idx_x = c_i_vec + i
                n_idx_y = c_j_vec + j
                n_idx_z = c_k_vec + k

                # --- PBC CHECK (vectorized) ---
                if use_periodic:
                    # Wrap around
                    n_idx_x = (n_idx_x + n_cells_x) % n_cells_x
                    n_idx_y = (n_idx_y + n_cells_y) % n_cells_y
                    n_idx_z = (n_idx_z + n_cells_z) % n_cells_z
                    valid_cell_mask = mask_m  # All cells valid with PBC
                else:
                    # Strict Bounds Check (vectorized)
                    valid_cell_mask = (
                        (n_idx_x >= 0)
                        & (n_idx_x < n_cells_x)
                        & (n_idx_y >= 0)
                        & (n_idx_y < n_cells_y)
                        & (n_idx_z >= 0)
                        & (n_idx_z < n_cells_z)
                        & mask_m
                    )

                # Compute flat cell indices (vectorized, one per query atom)
                cell_idx_vec = (
                    n_idx_x * (n_cells_y * n_cells_z) + n_idx_y * n_cells_z + n_idx_z
                )

                # Gather cell boundaries for each query atom's neighbor cell
                start_vec = tl.load(
                    CellStart + cell_idx_vec, mask=valid_cell_mask, other=-1
                )
                end_vec = tl.load(CellEnd + cell_idx_vec, mask=valid_cell_mask, other=0)

                # Find the maximum range we need to scan
                max_end = tl.max(tl.where(valid_cell_mask, end_vec, 0))
                min_start = tl.min(
                    tl.where(valid_cell_mask & (start_vec >= 0), start_vec, n_atoms)
                )

                # Use fixed iteration count for CUDA graph compatibility
                # Iterate over all possible neighbor atoms (worst case: all atoms)
                for tile_idx in range(MAX_TILES):
                    # Neighbor atom indices (global indices in sorted array)
                    idx_n = tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                    # Only process if within valid range [min_start, max_end) and within n_atoms
                    mask_n = (
                        (idx_n >= min_start) & (idx_n < max_end) & (idx_n < n_atoms)
                    )

                    # Load Candidate atoms
                    off_n = idx_n
                    pos_nx = tl.load(
                        SortedCoords + off_n * 3 + 0, mask=mask_n, other=0.0
                    )
                    pos_ny = tl.load(
                        SortedCoords + off_n * 3 + 1, mask=mask_n, other=0.0
                    )
                    pos_nz = tl.load(
                        SortedCoords + off_n * 3 + 2, mask=mask_n, other=0.0
                    )
                    batch_n = tl.load(SortedBatch + off_n, mask=mask_n, other=-2)

                    # Compute distances: [BLOCK_M query atoms, BLOCK_M candidate atoms]
                    d_x = pos_mx[:, None] - pos_nx[None, :]
                    d_y = pos_my[:, None] - pos_ny[None, :]
                    d_z = pos_mz[:, None] - pos_nz[None, :]

                    # --- PBC Distance Adjustment ---
                    if use_periodic:
                        d_x = d_x - box_x * _tl_round(d_x / box_x)
                        d_y = d_y - box_y * _tl_round(d_y / box_y)
                        d_z = d_z - box_z * _tl_round(d_z / box_z)

                    dist_sq = d_x * d_x + d_y * d_y + d_z * d_z

                    # --- Interaction Logic ---
                    # 1. Distance & Batch conditions
                    cond_dist = (dist_sq <= cutoff_upper_sq) & (
                        dist_sq >= cutoff_lower_sq
                    )
                    cond_batch = batch_m[:, None] == batch_n[None, :]

                    # 2. Check if neighbor is within each query atom's cell range
                    # idx_n is [BLOCK_M], start_vec is [BLOCK_M], end_vec is [BLOCK_M]
                    # We need [BLOCK_M, BLOCK_M] comparison
                    idx_n_bc = idx_n[None, :]  # [1, BLOCK_M]
                    start_bc = start_vec[:, None]  # [BLOCK_M, 1]
                    end_bc = end_vec[:, None]  # [BLOCK_M, 1]
                    valid_cell_bc = valid_cell_mask[:, None]  # [BLOCK_M, 1]

                    # Neighbor must be in the query atom's specific cell range
                    cond_in_range = (
                        (idx_n_bc >= start_bc) & (idx_n_bc < end_bc) & valid_cell_bc
                    )

                    # Load original indices for loop/transpose logic
                    # The condition must be based on original indices, not sorted indices
                    orig_m = tl.load(SortedToOrig + idx_m, mask=mask_m, other=-1)
                    orig_n = tl.load(SortedToOrig + idx_n, mask=mask_n, other=-1)

                    # 3. Index / Loop Logic (using original indices)
                    orig_m_bc = orig_m[:, None]
                    orig_n_bc = orig_n[None, :]

                    if include_transpose:
                        # Full List (both i->j and j->i)
                        if loop:
                            # Accept everything (i!=j AND i==j) -> All True
                            cond_idx = True
                        else:
                            # Accept only i != j
                            cond_idx = orig_m_bc != orig_n_bc
                    else:
                        # Half List (lower triangle convention: i >= j)
                        if loop:
                            # Accept i >= j (includes diagonal)
                            cond_idx = orig_m_bc >= orig_n_bc
                        else:
                            # Accept i > j (excludes diagonal)
                            cond_idx = orig_m_bc > orig_n_bc

                    valid_mask = (
                        cond_dist
                        & cond_batch
                        & cond_idx
                        & cond_in_range
                        & mask_m[:, None]
                        & mask_n[None, :]
                    )

                    # --- Store Block ---
                    num_found = tl.sum(valid_mask.to(tl.int32))
                    if num_found > 0:
                        current_offset = tl.atomic_add(GlobalCounter, num_found)
                        if current_offset + num_found <= max_pairs:
                            flat_mask = tl.ravel(valid_mask)
                            csum = tl.cumsum(flat_mask.to(tl.int32), axis=0)
                            store_idx = current_offset + csum - 1

                            flat_orig_m = tl.ravel(
                                tl.broadcast_to(orig_m[:, None], (BLOCK_M, BLOCK_M))
                            )
                            flat_orig_n = tl.ravel(
                                tl.broadcast_to(orig_n[None, :], (BLOCK_M, BLOCK_M))
                            )

                            tl.store(
                                OutPairs + 0 * max_pairs + store_idx,
                                flat_orig_m,
                                mask=flat_mask,
                            )
                            tl.store(
                                OutPairs + 1 * max_pairs + store_idx,
                                flat_orig_n,
                                mask=flat_mask,
                            )
                            tl.store(
                                OutDeltas + store_idx * 3 + 0,
                                tl.ravel(d_x),
                                mask=flat_mask,
                            )
                            tl.store(
                                OutDeltas + store_idx * 3 + 1,
                                tl.ravel(d_y),
                                mask=flat_mask,
                            )
                            tl.store(
                                OutDeltas + store_idx * 3 + 2,
                                tl.ravel(d_z),
                                mask=flat_mask,
                            )
                            tl.store(
                                OutDists + store_idx,
                                tl.sqrt(tl.ravel(dist_sq)),
                                mask=flat_mask,
                            )


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


class TritonCellNeighborAutograd(TritonNeighborAutograd):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        positions: Tensor,
        batch: Tensor,
        box_vectors: Tensor,
        use_periodic: bool,
        cutoff_lower: torch.float32,
        cutoff_upper: torch.float32,
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
        if max_num_pairs <= 0:
            raise ValueError('Expected "max_num_pairs" to be positive')

        # Handle box_vectors - following CUDA cell implementation approach:
        # The box must be provided and valid. For non-periodic, the box defines the cell grid bounds.
        # Move to correct device/dtype only if needed (avoid CPU sync during graph capture)
        # if box_vectors.device != device or box_vectors.dtype != dtype:
        #     box_vectors = box_vectors.to(dtype=dtype, device=device)
        box_vectors = box_vectors.contiguous()
        box_diag = box_vectors[0] if box_vectors.dim() == 3 else box_vectors
        # Keep as tensors to avoid GPU->CPU sync during CUDA graph capture
        box_x = box_diag[0, 0]
        box_y = box_diag[1, 1]
        box_z = box_diag[2, 2]
        # Create a contiguous tensor for kernel (CUDA graph compatible)
        box_sizes = torch.stack([box_x, box_y, box_z]).contiguous()

        # Compute dimensions using torch operations (stays on GPU)
        cell_dims = _get_cell_dimensions(box_x, box_y, box_z, cutoff_upper)

        # 1. Cell Index Calculation
        # For periodic: wrap positions to [0, box) using PBC
        # For non-periodic: shift positions by half box (matching CUDA cell implementation)
        # The CUDA implementation uses: (p + 0.5 * box_size) / cutoff for cell index
        if use_periodic:
            inv_box_x = 1.0 / box_x
            inv_box_y = 1.0 / box_y
            inv_box_z = 1.0 / box_z
            px = positions[:, 0] - torch.floor(positions[:, 0] * inv_box_x) * box_x
            py = positions[:, 1] - torch.floor(positions[:, 1] * inv_box_y) * box_y
            pz = positions[:, 2] - torch.floor(positions[:, 2] * inv_box_z) * box_z
        else:
            # For non-periodic, shift positions by half box (like CUDA impl)
            # This centers the cell grid around the origin
            px = positions[:, 0] + 0.5 * box_x
            py = positions[:, 1] + 0.5 * box_y
            pz = positions[:, 2] + 0.5 * box_z

        # Compute cell indices and clamp using broadcasting to avoid scalar extraction
        # Stack positions for vectorized operation
        p_stacked = torch.stack([px, py, pz], dim=1)  # (n_atoms, 3)
        cell_coords = (p_stacked / cutoff_upper).long()  # (n_atoms, 3)

        # Clamp using broadcasting: cell_dims is shape (3,), cell_coords is (n_atoms, 3)
        cell_coords = torch.clamp(
            cell_coords, min=torch.zeros(3, device=device), max=cell_dims - 1
        )

        # Extract individual coordinates
        cx = cell_coords[:, 0]
        cy = cell_coords[:, 1]
        cz = cell_coords[:, 2]

        cell_indices = cx * (cell_dims[1] * cell_dims[2]) + cy * cell_dims[2] + cz

        # 2. Sort
        sorted_cell_indices, sorted_atom_indices = torch.sort(cell_indices)
        sorted_atom_indices = sorted_atom_indices.int()

        sorted_positions = positions.index_select(
            0, sorted_atom_indices.long()
        ).contiguous()
        sorted_batch = batch.index_select(0, sorted_atom_indices.long()).contiguous()

        # 3. Cell Pointers (CUDA graph compatible using scatter_reduce)
        # Initialize cell_start with n_atoms (will use scatter_reduce 'amin')
        # Initialize cell_end with 0 (will use scatter_reduce 'amax')
        # num_cells_t is a tensor, PyTorch will extract the value for buffer size
        cell_start = torch.full((num_cells,), n_atoms, device=device, dtype=torch.int32)
        cell_end = torch.zeros((num_cells,), device=device, dtype=torch.int32)

        # Compute cell boundaries using scatter_reduce (fixed output size, CUDA graph compatible)
        atom_indices = torch.arange(n_atoms, device=device, dtype=torch.int32)
        sorted_cell_indices_long = sorted_cell_indices.long()
        # cell_start[cell] = min index of atoms in that cell
        cell_start.scatter_reduce_(
            0, sorted_cell_indices_long, atom_indices, reduce="amin", include_self=True
        )
        # cell_end[cell] = max index + 1 of atoms in that cell
        end_indices = atom_indices + 1
        cell_end.scatter_reduce_(
            0, sorted_cell_indices_long, end_indices, reduce="amax", include_self=True
        )

        # Mark empty cells with -1 for cell_start (cells where cell_end remained 0)
        # Use in-place operation to avoid tensor creation during CUDA graph capture
        empty_cell_mask = cell_end == 0
        cell_start.masked_fill_(empty_cell_mask, -1)

        # 4. Kernel Launch
        # Initialize neighbors to -1 so unused slots are properly marked
        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        BLOCK_M = 64
        grid = (triton.cdiv(n_atoms, BLOCK_M),)

        # Compute MAX_TILES for fixed iteration count (CUDA graph compatible)
        max_tiles = (n_atoms + BLOCK_M - 1) // BLOCK_M

        neighbor_list_kernel[grid](
            sorted_positions,
            sorted_batch,
            sorted_atom_indices,
            cell_start,
            cell_end,
            neighbors,
            deltas,
            distances,
            counter,
            box_sizes,
            cell_dims,
            n_atoms,
            max_num_pairs,
            cutoff_lower**2,
            cutoff_upper**2,
            use_periodic=use_periodic,
            loop=loop,
            include_transpose=include_transpose,
            BLOCK_M=BLOCK_M,
            MAX_TILES=max_tiles,
        )

        # Return counter as tensor to avoid GPU->CPU sync (CUDA graph compatible)
        num_pairs = counter.to(torch.long)

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        bwd = TritonNeighborAutograd.backward(
            ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs
        )
        return *bwd, None
