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
from torchmdnet.extensions.triton_neighbors import _tl_round, TritonNeighborAutograd


@triton.jit
def _neighbor_shared_kernel(
    pos_ptr,
    batch_ptr,
    box_ptr,
    neighbors0_ptr,
    neighbors1_ptr,
    deltas_ptr,
    distances_ptr,
    counter_ptr,
    stride_pos_0: tl.constexpr,
    stride_pos_1: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_box_b: tl.constexpr,
    stride_box_r: tl.constexpr,
    stride_box_c: tl.constexpr,
    n_atoms: tl.constexpr,
    use_periodic: tl.constexpr,
    include_transpose: tl.constexpr,
    loop: tl.constexpr,
    max_pairs: tl.constexpr,
    cutoff_lower2: tl.constexpr,
    cutoff_upper2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Shared-memory style neighbor construction: tile j, one block of i atoms."""
    pid_i = tl.program_id(axis=0)
    pid_tile = tl.program_id(axis=1)

    i_idx = pid_i * BLOCK + tl.arange(0, BLOCK)
    j_idx = pid_tile * BLOCK + tl.arange(0, BLOCK)

    mask_i = i_idx < n_atoms
    mask_j = j_idx < n_atoms

    batch_i = tl.load(batch_ptr + i_idx * stride_batch, mask=mask_i, other=0)
    batch_j = tl.load(batch_ptr + j_idx * stride_batch, mask=mask_j, other=0)

    pos_ix = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 0 * stride_pos_1, mask=mask_i, other=0.0
    )
    pos_iy = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 1 * stride_pos_1, mask=mask_i, other=0.0
    )
    pos_iz = tl.load(
        pos_ptr + i_idx * stride_pos_0 + 2 * stride_pos_1, mask=mask_i, other=0.0
    )

    pos_jx = tl.load(
        pos_ptr + j_idx * stride_pos_0 + 0 * stride_pos_1, mask=mask_j, other=0.0
    )
    pos_jy = tl.load(
        pos_ptr + j_idx * stride_pos_0 + 1 * stride_pos_1, mask=mask_j, other=0.0
    )
    pos_jz = tl.load(
        pos_ptr + j_idx * stride_pos_0 + 2 * stride_pos_1, mask=mask_j, other=0.0
    )

    # Pairwise masks (BLOCK, BLOCK)
    mask_pair = mask_i[:, None] & mask_j[None, :]
    if loop:
        mask_pair = mask_pair & (
            (j_idx[None, :] < i_idx[:, None]) | (j_idx[None, :] == i_idx[:, None])
        )
    else:
        mask_pair = mask_pair & (j_idx[None, :] < i_idx[:, None])
    mask_pair = mask_pair & (batch_i[:, None] == batch_j[None, :])

    dx = pos_ix[:, None] - pos_jx[None, :]
    dy = pos_iy[:, None] - pos_jy[None, :]
    dz = pos_iz[:, None] - pos_jz[None, :]

    if use_periodic:
        box_base = box_ptr + batch_i * stride_box_b
        b20 = tl.load(
            box_base + 2 * stride_box_r + 0 * stride_box_c, mask=mask_i, other=0.0
        )
        b21 = tl.load(
            box_base + 2 * stride_box_r + 1 * stride_box_c, mask=mask_i, other=0.0
        )
        b22 = tl.load(
            box_base + 2 * stride_box_r + 2 * stride_box_c, mask=mask_i, other=1.0
        )
        b10 = tl.load(
            box_base + 1 * stride_box_r + 0 * stride_box_c, mask=mask_i, other=0.0
        )
        b11 = tl.load(
            box_base + 1 * stride_box_r + 1 * stride_box_c, mask=mask_i, other=1.0
        )
        b00 = tl.load(
            box_base + 0 * stride_box_r + 0 * stride_box_c, mask=mask_i, other=1.0
        )

        scale3 = _tl_round(dz / b22[:, None])
        dx = dx - scale3 * b20[:, None]
        dy = dy - scale3 * b21[:, None]
        dz = dz - scale3 * b22[:, None]

        scale2 = _tl_round(dy / b11[:, None])
        dx = dx - scale2 * b10[:, None]
        dy = dy - scale2 * b11[:, None]

        scale1 = _tl_round(dx / b00[:, None])
        dx = dx - scale1 * b00[:, None]

    dist2 = dx * dx + dy * dy + dz * dz
    valid_pair = mask_pair & (dist2 < cutoff_upper2) & (dist2 >= cutoff_lower2)

    i_mat = i_idx[:, None] + tl.zeros((1, BLOCK), dtype=i_idx.dtype)
    j_mat = tl.zeros((BLOCK, 1), dtype=j_idx.dtype) + j_idx[None, :]

    dx_flat = tl.reshape(dx, (BLOCK * BLOCK,))
    dy_flat = tl.reshape(dy, (BLOCK * BLOCK,))
    dz_flat = tl.reshape(dz, (BLOCK * BLOCK,))
    dist_flat = tl.reshape(tl.sqrt(dist2), (BLOCK * BLOCK,))

    i_flat = tl.reshape(i_mat, (BLOCK * BLOCK,))
    j_flat = tl.reshape(j_mat, (BLOCK * BLOCK,))

    valid_flat = tl.reshape(valid_pair, (BLOCK * BLOCK,))
    valid_int = valid_flat.to(tl.int32)
    total_pairs = tl.sum(valid_int)

    if include_transpose:
        valid_t_flat = valid_flat & (i_flat != j_flat)
        valid_t_int = valid_t_flat.to(tl.int32)
        total_transpose = tl.sum(valid_t_int)
        total_out = total_pairs + total_transpose
    else:
        total_out = total_pairs

    has_out = total_out > 0
    start_idx = tl.atomic_add(counter_ptr, total_out, mask=has_out)
    start_idx = tl.where(has_out, start_idx, 0)
    start_idx = tl.broadcast_to(start_idx, valid_flat.shape)

    prefix = tl.cumsum(valid_int, axis=0) - 1
    out_idx = start_idx + prefix

    mask_store = valid_flat & (out_idx < max_pairs)
    tl.store(neighbors0_ptr + out_idx, i_flat, mask=mask_store)
    tl.store(neighbors1_ptr + out_idx, j_flat, mask=mask_store)
    tl.store(deltas_ptr + out_idx * 3 + 0, dx_flat, mask=mask_store)
    tl.store(deltas_ptr + out_idx * 3 + 1, dy_flat, mask=mask_store)
    tl.store(deltas_ptr + out_idx * 3 + 2, dz_flat, mask=mask_store)
    tl.store(distances_ptr + out_idx, dist_flat, mask=mask_store)

    if include_transpose:
        prefix_t = tl.cumsum(valid_t_int, axis=0) - 1
        out_idx_t = start_idx + total_pairs + prefix_t

        mask_store_t = valid_t_flat & (out_idx_t < max_pairs)
        tl.store(neighbors0_ptr + out_idx_t, j_flat, mask=mask_store_t)
        tl.store(neighbors1_ptr + out_idx_t, i_flat, mask=mask_store_t)
        tl.store(deltas_ptr + out_idx_t * 3 + 0, -dx_flat, mask=mask_store_t)
        tl.store(deltas_ptr + out_idx_t * 3 + 1, -dy_flat, mask=mask_store_t)
        tl.store(deltas_ptr + out_idx_t * 3 + 2, -dz_flat, mask=mask_store_t)
        tl.store(distances_ptr + out_idx_t, dist_flat, mask=mask_store_t)


class TritonSharedNeighborAutograd(TritonNeighborAutograd):
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

        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError('Expected "positions" to have shape (N, 3)')
        if batch.dim() != 1 or batch.size(0) != n_atoms:
            raise ValueError('Expected "batch" to have shape (N,)')
        if max_num_pairs <= 0:
            raise ValueError('Expected "max_num_pairs" to be positive')

        batch = batch.contiguous()
        positions = positions.contiguous()

        if use_periodic:
            if box_vectors.dim() == 2:
                box_vectors = box_vectors.unsqueeze(0)
            elif box_vectors.dim() != 3:
                raise ValueError('Expected "box_vectors" to have shape (n_batch, 3, 3)')
            box_vectors = box_vectors.to(device=device, dtype=dtype).contiguous()
            box_stride_0 = 0 if box_vectors.size(0) == 1 else box_vectors.stride(0)
        else:
            # Dummy tensor to avoid conditional code paths in the kernel signature.
            box_vectors = positions
            box_stride_0 = 0

        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        BLOCK = 64
        num_blocks = max((n_atoms + BLOCK - 1) // BLOCK, 1)
        grid = (num_blocks, num_blocks)

        _neighbor_shared_kernel[grid](
            positions,
            batch,
            box_vectors,
            neighbors[0],
            neighbors[1],
            deltas,
            distances,
            counter,
            positions.stride(0),
            positions.stride(1),
            batch.stride(0),
            box_stride_0,
            box_vectors.stride(1) if use_periodic else 0,
            box_vectors.stride(2) if use_periodic else 0,
            n_atoms,
            use_periodic,
            include_transpose,
            loop,
            max_num_pairs,
            float(cutoff_lower * cutoff_lower),
            float(cutoff_upper * cutoff_upper),
            BLOCK=BLOCK,
        )

        num_pairs = counter.to(torch.long)
        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs
