try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

from torch import Tensor
import torch
from torchmdnet.extensions.triton_neighbors import (
    _tl_round,
    TritonNeighborAutograd,
    _validate_box,
    _get_cell_dimensions,
)


@triton.jit
def _assign_cell_indices_kernel(
    pos_ptr,
    cell_index_ptr,
    stride_pos_0: tl.constexpr,
    stride_pos_1: tl.constexpr,
    n_atoms: tl.constexpr,
    box_x: tl.constexpr,
    box_y: tl.constexpr,
    box_z: tl.constexpr,
    cutoff: tl.constexpr,
    cell_x: tl.constexpr,
    cell_y: tl.constexpr,
    cell_z: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_atoms

    px = tl.load(pos_ptr + idx * stride_pos_0 + 0 * stride_pos_1, mask=mask, other=0.0)
    py = tl.load(pos_ptr + idx * stride_pos_0 + 1 * stride_pos_1, mask=mask, other=0.0)
    pz = tl.load(pos_ptr + idx * stride_pos_0 + 2 * stride_pos_1, mask=mask, other=0.0)

    # Bring to [-0.5*box, 0.5*box] (rectangular PBC)
    px = px - _tl_round(px / box_x) * box_x
    py = py - _tl_round(py / box_y) * box_y
    pz = pz - _tl_round(pz / box_z) * box_z

    cx = tl.math.floor((px + 0.5 * box_x) / cutoff).to(tl.int32)
    cy = tl.math.floor((py + 0.5 * box_y) / cutoff).to(tl.int32)
    cz = tl.math.floor((pz + 0.5 * box_z) / cutoff).to(tl.int32)

    cx = tl.where(cx == cell_x, 0, cx)
    cy = tl.where(cy == cell_y, 0, cy)
    cz = tl.where(cz == cell_z, 0, cz)

    cell_index = cx + cell_x * (cy + cell_y * cz)
    tl.store(cell_index_ptr + idx, cell_index, mask=mask)


@triton.jit
def _fill_cell_offsets_kernel(
    sorted_cell_ptr,
    cell_start_ptr,
    cell_end_ptr,
    n_atoms: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_atoms

    icell = tl.load(sorted_cell_ptr + idx, mask=mask, other=0)
    prev = tl.load(sorted_cell_ptr + idx - 1, mask=mask & (idx > 0), other=0)

    is_first = idx == 0
    is_new = is_first | (icell != prev)

    tl.store(cell_start_ptr + icell, idx, mask=mask & is_new)
    tl.store(cell_end_ptr + prev, idx, mask=mask & (is_new & (idx > 0)))
    tl.store(cell_end_ptr + icell, idx + 1, mask=mask & (idx == n_atoms - 1))


@triton.jit
def _traverse_cell_kernel(
    pos_ptr,
    batch_ptr,
    sorted_index_ptr,
    cell_start_ptr,
    cell_end_ptr,
    neighbors0_ptr,
    neighbors1_ptr,
    deltas_ptr,
    distances_ptr,
    counter_ptr,
    stride_pos_0: tl.constexpr,
    stride_pos_1: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_sorted: tl.constexpr,
    n_atoms: tl.constexpr,
    num_cells: tl.constexpr,
    cell_x: tl.constexpr,
    cell_y: tl.constexpr,
    cell_z: tl.constexpr,
    box_x: tl.constexpr,
    box_y: tl.constexpr,
    box_z: tl.constexpr,
    cutoff: tl.constexpr,
    cutoff_lower2: tl.constexpr,
    cutoff_upper2: tl.constexpr,
    use_periodic: tl.constexpr,
    include_transpose: tl.constexpr,
    loop: tl.constexpr,
    max_pairs: tl.constexpr,
    BLOCK_ATOMS: tl.constexpr,
    BLOCK_NEI: tl.constexpr,
    MAX_CELL_ITERS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i_idx = pid * BLOCK_ATOMS + tl.arange(0, BLOCK_ATOMS)
    mask_i = i_idx < n_atoms

    pos_ix = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 0 * stride_pos_1, mask=mask_i, other=0.0
    )
    pos_iy = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 1 * stride_pos_1, mask=mask_i, other=0.0
    )
    pos_iz = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 2 * stride_pos_1, mask=mask_i, other=0.0
    )

    batch_i = tl.load(batch_ptr + i_idx * stride_batch, mask=mask_i, other=0)

    # Cell coordinate for atom i
    pwx = pos_ix - _tl_round(pos_ix / box_x) * box_x
    pwy = pos_iy - _tl_round(pos_iy / box_y) * box_y
    pwz = pos_iz - _tl_round(pos_iz / box_z) * box_z
    cx = tl.math.floor((pwx + 0.5 * box_x) / cutoff).to(tl.int32)
    cy = tl.math.floor((pwy + 0.5 * box_y) / cutoff).to(tl.int32)
    cz = tl.math.floor((pwz + 0.5 * box_z) / cutoff).to(tl.int32)
    cx = tl.where(cx == cell_x, 0, cx)
    cy = tl.where(cy == cell_y, 0, cy)
    cz = tl.where(cz == cell_z, 0, cz)

    offs_nei = tl.arange(0, BLOCK_NEI)

    counter_broadcast = counter_ptr + tl.zeros_like(i_idx)

    for neigh in range(27):
        dx_cell = (neigh % 3) - 1
        dy_cell = ((neigh // 3) % 3) - 1
        dz_cell = (neigh // 9) - 1

        nx = cx + dx_cell
        ny = cy + dy_cell
        nz = cz + dz_cell

        nx = tl.where(nx < 0, nx + cell_x, nx)
        ny = tl.where(ny < 0, ny + cell_y, ny)
        nz = tl.where(nz < 0, nz + cell_z, nz)
        nx = tl.where(nx >= cell_x, nx - cell_x, nx)
        ny = tl.where(ny >= cell_y, ny - cell_y, ny)
        nz = tl.where(nz >= cell_z, nz - cell_z, nz)

        neighbor_cell = nx + cell_x * (ny + cell_y * nz)
        start = tl.load(cell_start_ptr + neighbor_cell, mask=mask_i, other=-1)
        end = tl.load(cell_end_ptr + neighbor_cell, mask=mask_i, other=0)
        has_atoms = start != -1

        for it in range(MAX_CELL_ITERS):
            base = start + it * BLOCK_NEI
            j_idx = base + offs_nei
            valid_j = mask_i & has_atoms & (j_idx < end)
            if loop:
                valid_j = valid_j & ((j_idx < i_idx) | (j_idx == i_idx))
            else:
                valid_j = valid_j & (j_idx < i_idx)

            batch_j = tl.load(batch_ptr + j_idx * stride_batch, mask=valid_j, other=0)
            valid_j = valid_j & (batch_j == batch_i)

            pos_jx = tl.load(
                pos_ptr + j_idx * stride_pos_0 + 0 * stride_pos_1,
                mask=valid_j,
                other=0.0,
            )
            pos_jy = tl.load(
                pos_ptr + j_idx * stride_pos_0 + 1 * stride_pos_1,
                mask=valid_j,
                other=0.0,
            )
            pos_jz = tl.load(
                pos_ptr + j_idx * stride_pos_0 + 2 * stride_pos_1,
                mask=valid_j,
                other=0.0,
            )

            dx = pos_ix - pos_jx
            dy = pos_iy - pos_jy
            dz = pos_iz - pos_jz

            if use_periodic:
                dx = dx - _tl_round(dx / box_x) * box_x
                dy = dy - _tl_round(dy / box_y) * box_y
                dz = dz - _tl_round(dz / box_z) * box_z

            dist2 = dx * dx + dy * dy + dz * dz
            is_self = j_idx == i_idx
            valid_pair = valid_j & (
                ((dist2 < cutoff_upper2) & (dist2 >= cutoff_lower2)) | (loop & is_self)
            )

            ori_i = tl.load(
                sorted_index_ptr + i_idx * stride_sorted, mask=mask_i, other=0
            )
            orj = tl.load(
                sorted_index_ptr + j_idx * stride_sorted, mask=valid_j, other=0
            )
            ori = ori_i

            # Elementwise max/min; tl.max would perform a reduction over an axis
            ni = tl.maximum(ori, orj)
            nj = tl.minimum(ori, orj)
            sign = tl.where(ni == ori, 1.0, -1.0).to(dx.dtype)

            dx_out = dx * sign
            dy_out = dy * sign
            dz_out = dz * sign
            dist = tl.sqrt(dist2)

            # Reserve contiguous slots for valid pairs in this block (BLOCK_ATOMS=1 -> 1D)
            valid_pair_int = valid_pair.to(tl.int32)
            total_pairs = tl.sum(valid_pair_int)
            base_idx = tl.atomic_add(counter_ptr, total_pairs, mask=total_pairs > 0)
            local_idx = tl.cumsum(valid_pair_int, axis=0) - 1
            out_idx = base_idx + local_idx

            mask_store = valid_pair & (out_idx < max_pairs)
            tl.store(neighbors0_ptr + out_idx, ni, mask=mask_store)
            tl.store(neighbors1_ptr + out_idx, nj, mask=mask_store)
            tl.store(deltas_ptr + out_idx * 3 + 0, dx_out, mask=mask_store)
            tl.store(deltas_ptr + out_idx * 3 + 1, dy_out, mask=mask_store)
            tl.store(deltas_ptr + out_idx * 3 + 2, dz_out, mask=mask_store)
            tl.store(distances_ptr + out_idx, dist, mask=mask_store)

            if include_transpose:
                valid_t = valid_pair & (ni != nj)
                valid_t_int = valid_t.to(tl.int32)
                total_pairs_t = tl.sum(valid_t_int)
                base_idx_t = tl.atomic_add(
                    counter_ptr, total_pairs_t, mask=total_pairs_t > 0
                )
                local_idx_t = tl.cumsum(valid_t_int, axis=0) - 1
                out_idx_t = base_idx_t + local_idx_t

                mask_store_t = valid_t & (out_idx_t < max_pairs)
                tl.store(neighbors0_ptr + out_idx_t, nj, mask=mask_store_t)
                tl.store(neighbors1_ptr + out_idx_t, ni, mask=mask_store_t)
                tl.store(deltas_ptr + out_idx_t * 3 + 0, -dx_out, mask=mask_store_t)
                tl.store(deltas_ptr + out_idx_t * 3 + 1, -dy_out, mask=mask_store_t)
                tl.store(deltas_ptr + out_idx_t * 3 + 2, -dz_out, mask=mask_store_t)
                tl.store(distances_ptr + out_idx_t, dist, mask=mask_store_t)


class TritonCellNeighborAutograd(TritonNeighborAutograd):
    @staticmethod
    def forward(  # type: ignore[override]
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
    ):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not available")
        device = positions.device
        dtype = positions.dtype
        n_atoms = positions.size(0)

        # Validate inputs (mirror CUDA constraints)
        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError('Expected "positions" to have shape (N, 3)')
        if batch.dim() != 1 or batch.size(0) != n_atoms:
            raise ValueError('Expected "batch" to have shape (N,)')
        if max_num_pairs <= 0:
            raise ValueError('Expected "max_num_pairs" to be positive')

        # Ensure box is well-formed and diagonal (cell list only supports rectangular boxes)
        box_vectors = _validate_box(box_vectors, float(cutoff_upper), 1)
        box_vectors = box_vectors.to(device=device, dtype=dtype).contiguous()
        # CUDA version squeezes (1, 3, 3) to (3, 3); do the same to avoid OOB indexing
        box_diag = box_vectors[0] if box_vectors.dim() == 3 else box_vectors

        # Cell grid dimensions (host math to match CUDA getCellDimensions)
        cell_dim_x, cell_dim_y, cell_dim_z = _get_cell_dimensions(
            box_vectors, float(cutoff_upper)
        )
        num_cells = cell_dim_x * cell_dim_y * cell_dim_z

        batch = batch.contiguous()
        positions = positions.contiguous()

        # 1. Compute cell index per atom
        cell_indices = torch.empty((n_atoms,), device=device, dtype=torch.int32)
        grid_assign = lambda meta: (triton.cdiv(n_atoms, meta["BLOCK"]),)
        _assign_cell_indices_kernel[grid_assign](
            positions,
            cell_indices,
            positions.stride(0),
            positions.stride(1),
            n_atoms,
            float(box_diag[0, 0]),
            float(box_diag[1, 1]),
            float(box_diag[2, 2]),
            float(cutoff_upper),
            cell_dim_x,
            cell_dim_y,
            cell_dim_z,
            BLOCK=256,
        )

        # 2. Sort atoms by cell index
        sorted_cell_indices, sorted_atom_indices = torch.sort(cell_indices)
        sorted_atom_indices = sorted_atom_indices.to(torch.int32)
        sorted_positions = positions.index_select(0, sorted_atom_indices)
        sorted_batch = batch.index_select(0, sorted_atom_indices)

        # 3. Build cell start/end offsets
        cell_start = torch.full((num_cells,), -1, device=device, dtype=torch.int32)
        cell_end = torch.empty((num_cells,), device=device, dtype=torch.int32)
        grid_offsets = lambda meta: (triton.cdiv(n_atoms, meta["BLOCK"]),)
        _fill_cell_offsets_kernel[grid_offsets](
            sorted_cell_indices,
            cell_start,
            cell_end,
            n_atoms,
            BLOCK=256,
        )

        # 4. Traverse cell list to generate neighbor pairs
        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        grid_traverse = lambda meta: (triton.cdiv(n_atoms, meta["BLOCK_ATOMS"]),)
        _traverse_cell_kernel[grid_traverse](
            sorted_positions,
            sorted_batch,
            sorted_atom_indices,
            cell_start,
            cell_end,
            neighbors[0],
            neighbors[1],
            deltas,
            distances,
            counter,
            sorted_positions.stride(0),
            sorted_positions.stride(1),
            sorted_batch.stride(0),
            sorted_atom_indices.stride(0),
            n_atoms,
            num_cells,
            cell_dim_x,
            cell_dim_y,
            cell_dim_z,
            float(box_diag[0, 0]),
            float(box_diag[1, 1]),
            float(box_diag[2, 2]),
            float(cutoff_upper),
            float(cutoff_lower * cutoff_lower),
            float(cutoff_upper * cutoff_upper),
            use_periodic,
            include_transpose,
            loop,
            max_num_pairs,
            BLOCK_ATOMS=1,
            BLOCK_NEI=64,
            MAX_CELL_ITERS=16,
        )

        num_pairs = counter.to(torch.long)
        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs
