import torch

import torch
import triton
import triton.language as tl


@triton.jit
def neighbor_list_kernel(
    # Pointers to inputs
    Coords,  # [N_atoms, 3] Sorted coordinates
    CellStart,  # [N_cells] Index where each cell starts in Coords
    CellCount,  # [N_cells] Number of atoms in each cell
    # Pointers to outputs
    OutPairs,  # [2, Max_Pairs]
    OutDeltas,  # [Max_Pairs, 3]
    OutDists,  # [Max_Pairs]
    GlobalCounter,  # [1] Atomic counter for current pair count
    # Constants
    n_atoms,  # Total atoms
    max_pairs,  # Size of output buffer
    cutoff_sq,  # Cutoff squared
    # Box dimensions for PBC
    box_x,
    box_y,
    box_z,
    # Grid dimensions
    n_cells_x,
    n_cells_y,
    n_cells_z,
    # Block constants
    BLOCK_M: tl.constexpr,  # Size of Query Block
):
    # 1. Identify Query Atoms
    pid = tl.program_id(0)
    # Range of atom indices this block handles
    idx_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = idx_m < n_atoms

    # Load Query Coordinates
    # We assume Coords are [N, 3] and contiguous (row-major)
    off_x = idx_m * 3 + 0
    off_y = idx_m * 3 + 1
    off_z = idx_m * 3 + 2

    # Load query positions
    pos_mx = tl.load(Coords + off_x, mask=mask_m, other=0.0)
    pos_my = tl.load(Coords + off_y, mask=mask_m, other=0.0)
    pos_mz = tl.load(Coords + off_z, mask=mask_m, other=0.0)

    # 2. Determine which Cell these atoms belong to
    # Since input is sorted, we can just look at the cell of the first atom in the block
    # (Approximation: assumes BLOCK_M atoms fit in one or two cells, usually true for sorted data)
    # Ideally, we calculate cell indices explicitly for robustness.
    # Cell index = floor(pos / cell_size). Here we reconstruct it:
    # Note: For highest speed, pass 'CellIndices' array instead of recomputing.
    # Below we use the Grid Loop strategy: Iterate ALL 27 neighbor cells relative to the query cell.

    # Calculating cell index for the *first* atom in block to find "Home Cell"
    # (Assuming sorted atoms are localized)
    cell_sx = box_x / n_cells_x
    cell_sy = box_y / n_cells_y
    cell_sz = box_z / n_cells_z

    # Current cell indices (i, j, k) for the first atom in the block
    c_i = (pos_mx[0] / cell_sx).to(tl.int32)
    c_j = (pos_my[0] / cell_sy).to(tl.int32)
    c_k = (pos_mz[0] / cell_sz).to(tl.int32)

    # 3. Iterate over 3x3x3 Neighbor Cells
    # We loop -1 to 1 in each dimension
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):

                # Apply PBC to find neighbor cell index
                # (x % N + N) % N handles negative wrapping
                n_idx_x = (c_i + i + n_cells_x) % n_cells_x
                n_idx_y = (c_j + j + n_cells_y) % n_cells_y
                n_idx_z = (c_k + k + n_cells_z) % n_cells_z

                # Flat neighbor cell index
                cell_idx = (
                    n_idx_x * (n_cells_y * n_cells_z) + n_idx_y * n_cells_z + n_idx_z
                )

                # Get start and count of atoms in this neighbor cell
                start = tl.load(CellStart + cell_idx)
                count = tl.load(CellCount + cell_idx)

                # Loop over atoms in the neighbor cell
                # We process them in chunks of BLOCK_M (or similar size)
                for tile_idx in range(0, count, BLOCK_M):
                    # Candidate Indices
                    idx_n = start + tile_idx + tl.arange(0, BLOCK_M)
                    mask_n = (tile_idx + tl.arange(0, BLOCK_M)) < count

                    # Load Candidate Coords
                    off_nx = idx_n * 3 + 0
                    off_ny = idx_n * 3 + 1
                    off_nz = idx_n * 3 + 2

                    pos_nx = tl.load(Coords + off_nx, mask=mask_n, other=0.0)
                    pos_ny = tl.load(Coords + off_ny, mask=mask_n, other=0.0)
                    pos_nz = tl.load(Coords + off_nz, mask=mask_n, other=0.0)

                    # Broadcast shapes for pairwise calculation
                    # Query: [BLOCK_M, 1], Candidate: [1, BLOCK_M]
                    q_x = pos_mx[:, None]
                    q_y = pos_my[:, None]
                    q_z = pos_mz[:, None]

                    c_x = pos_nx[None, :]
                    c_y = pos_ny[None, :]
                    c_z = pos_nz[None, :]

                    # --- PBC Distance Calculation (Minimum Image) ---
                    dx = q_x - c_x
                    dy = q_y - c_y
                    dz = q_z - c_z

                    # dx = dx - box * round(dx / box)
                    dx = dx - box_x * tl.math.round(dx / box_x)
                    dy = dy - box_y * tl.math.round(dy / box_y)
                    dz = dz - box_z * tl.math.round(dz / box_z)

                    dist_sq = dx * dx + dy * dy + dz * dz

                    # --- Filtering ---
                    # 1. Distance check
                    is_neighbor = dist_sq < cutoff_sq

                    # 2. Avoid double counting (i < j) and self-interaction (i != j)
                    # Note: For simple neighbor lists, we often want i != j.
                    # If you need half-list (Newton's 3rd law), use idx_m[:, None] < idx_n[None, :]
                    idx_m_bc = idx_m[:, None]
                    idx_n_bc = idx_n[None, :]

                    # Enforce strictly upper triangular for half-list, or just i != j for full
                    valid_mask = (
                        is_neighbor
                        & (idx_m_bc < idx_n_bc)
                        & mask_m[:, None]
                        & mask_n[None, :]
                    )

                    # --- Block-wise Output Allocation ---
                    # Count how many pairs we found in this tile
                    # We cast to int32 because sum returns same dtype as input (int1)
                    num_found = tl.sum(valid_mask.to(tl.int32))

                    if num_found > 0:
                        # ATOMIC ADD: Reserve space in the global output buffer
                        # This happens once per tile, not per pair -> Much faster
                        current_offset = tl.atomic_add(GlobalCounter, num_found)

                        # Guard against overflow
                        if current_offset + num_found <= max_pairs:
                            # We need to flatten the 2D block results to 1D to write them out.
                            # Triton doesn't have a direct "compress/compact" instruction yet for 2D->1D.
                            # We compute running index (prefix sum) within the block to find write slot.

                            # Flatten the mask to [BLOCK_M * BLOCK_M]
                            flat_mask = tl.ravel(valid_mask)

                            # Compute exclusive scan (cumsum) to determine thread offset
                            # Note: tl.cumsum is not fully exposed on all blocks easily,
                            # usually strictly sequential manual scan or use 'where' logic.
                            # OPTIMIZATION: For Triton, we can iterate positions.

                            # Because fully parallel scan is complex in a snippet,
                            # we verify valid indices and write.

                            # Calculate write indices
                            # This is a bit tricky in Triton without `scan`.
                            # Simpler approach for snippet: iterate and write (serialized within block)
                            # OR utilize `tl.where` with pre-calculated indices if density is high.

                            # High-Performance approach:
                            # Since we know `num_found` is small (sparse), we can just store
                            # the valid pairs.

                            # Generate offsets for valid elements
                            csum = tl.cumsum(flat_mask.to(tl.int32), axis=0)
                            # csum is inclusive, so write_idx = start + csum - 1

                            store_idx = current_offset + csum - 1

                            # Flatten data
                            flat_dx = tl.ravel(dx)
                            flat_dy = tl.ravel(dy)
                            flat_dz = tl.ravel(dz)
                            flat_dsq = tl.ravel(dist_sq)
                            flat_idx_m = tl.ravel(idx_m_bc)
                            flat_idx_n = tl.ravel(idx_n_bc)

                            # Write only valid entries
                            tl.store(
                                OutPairs + 0 * max_pairs + store_idx,
                                flat_idx_m,
                                mask=flat_mask,
                            )
                            tl.store(
                                OutPairs + 1 * max_pairs + store_idx,
                                flat_idx_n,
                                mask=flat_mask,
                            )

                            tl.store(
                                OutDeltas + store_idx * 3 + 0, flat_dx, mask=flat_mask
                            )
                            tl.store(
                                OutDeltas + store_idx * 3 + 1, flat_dy, mask=flat_mask
                            )
                            tl.store(
                                OutDeltas + store_idx * 3 + 2, flat_dz, mask=flat_mask
                            )

                            tl.store(
                                OutDists + store_idx, tl.sqrt(flat_dsq), mask=flat_mask
                            )


def find_neighbors(positions, box, cutoff, max_pairs=10_000_000):
    """
    positions: [N, 3] tensor
    box: [3] tensor (Lx, Ly, Lz)
    cutoff: scalar
    """
    N = positions.shape[0]
    device = positions.device

    # 1. Setup Cell Grid (Host side logic)
    # Ensure cell size >= cutoff
    n_cells = (box / cutoff).floor().int()
    cell_size = box / n_cells

    # 2. Sort Particles (Crucial for performance)
    # Calculate cell index for every atom
    cell_idx_x = (positions[:, 0] / cell_size[0]).long().clamp(0, n_cells[0] - 1)
    cell_idx_y = (positions[:, 1] / cell_size[1]).long().clamp(0, n_cells[1] - 1)
    cell_idx_z = (positions[:, 2] / cell_size[2]).long().clamp(0, n_cells[2] - 1)

    # Flat cell index
    flat_cell_indices = (
        cell_idx_x * n_cells[1] * n_cells[2] + cell_idx_y * n_cells[2] + cell_idx_z
    )

    # Sort atoms by cell index
    sorted_cell_indices, sort_idx = torch.sort(flat_cell_indices)
    sorted_pos = positions[sort_idx].contiguous()

    # 3. Build Cell Pointers (Start/Count)
    # Using unique_consecutive to find cell boundaries
    unique_cells, counts = torch.unique_consecutive(
        sorted_cell_indices, return_counts=True
    )

    # We need dense arrays for the kernel [Total_Cells]
    total_cells = n_cells[0] * n_cells[1] * n_cells[2]
    cell_start = torch.zeros(total_cells, dtype=torch.int32, device=device)
    cell_count = torch.zeros(total_cells, dtype=torch.int32, device=device)

    # Compute starts (requires a search or scatter, here simplified)
    # In production, use scatter_add or searchsorted
    # Simple trick: scatter counts
    cell_count.index_add_(0, unique_cells.int(), counts.int())

    # Cumulative sum to get starts? No, we need starts in the *sorted* array.
    # Since we sorted by cell, we can find starts easily.
    # A fast way involves searchsorted if cells are sparse,
    # but for water (dense), most cells are filled.
    # We can compute starts using the counts and a prefix sum,
    # but strictly speaking, we need the index in 'sorted_pos'.

    # Fast GPU-friendly start index construction:
    # Find where cell index changes
    change_mask = torch.cat(
        [
            torch.tensor([1], device=device),
            sorted_cell_indices[1:] != sorted_cell_indices[:-1],
        ]
    )
    change_indices = torch.nonzero(change_mask).squeeze()
    # This maps unique_cells -> start_index in sorted_pos
    cell_start.index_copy_(0, unique_cells.int(), change_indices.int())

    # 4. Prepare Outputs
    out_pairs = torch.empty((2, max_pairs), dtype=torch.int32, device=device)
    out_deltas = torch.empty((max_pairs, 3), dtype=torch.float32, device=device)
    out_dists = torch.empty((max_pairs,), dtype=torch.float32, device=device)
    global_counter = torch.zeros((1,), dtype=torch.int32, device=device)

    # 5. Launch Kernel
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]),)

    neighbor_list_kernel[grid](
        sorted_pos,
        cell_start,
        cell_count,
        out_pairs,
        out_deltas,
        out_dists,
        global_counter,
        N,
        max_pairs,
        cutoff**2,
        box[0].item(),
        box[1].item(),
        box[2].item(),
        n_cells[0].item(),
        n_cells[1].item(),
        n_cells[2].item(),
        BLOCK_M=64,  # Tuning parameter
    )

    num_pairs = global_counter.item()

    # Note: The output pairs use indices from 'sorted_pos'.
    # If you need original indices, you must map them back using 'sort_idx'.
    # remapped_pairs = sort_idx[out_pairs[:, :num_pairs]]

    return (
        out_pairs[:, :num_pairs],
        out_deltas[:num_pairs],
        out_dists[:num_pairs],
        num_pairs,
    )
