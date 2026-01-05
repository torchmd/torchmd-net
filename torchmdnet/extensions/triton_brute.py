# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
import triton
import triton.language as tl
from torch import Tensor
import torch
from torchmdnet.extensions.triton_neighbors import _tl_round, TritonNeighborAutograd
from torch.library import triton_op, wrap_triton
from typing import Tuple


@triton.jit
def _neighbor_brute_kernel(
    pos_ptr,
    batch_ptr,
    box_ptr,
    neighbors0_ptr,
    neighbors1_ptr,
    deltas_ptr,
    distances_ptr,
    counter_ptr,
    box_batch_stride,
    n_atoms,
    num_all_pairs,
    use_periodic: tl.constexpr,
    include_transpose: tl.constexpr,
    loop: tl.constexpr,
    max_pairs,
    cutoff_lower,
    cutoff_upper,
    BLOCK: tl.constexpr,
):
    """Brute-force neighbor list kernel using triangular indexing and atomic compaction.

    Uses triangular indexing to iterate over only n*(n-1)/2 pairs (or n*(n+1)/2 with loop),
    achieving 100% thread utilization while maintaining block-level atomic compaction.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)

    valid = idx < num_all_pairs

    # Convert linear index to (i, j) using triangular formula (same as CUDA get_row)
    # Do integer arithmetic first, only convert to float for sqrt
    if loop:
        # With self-loops: j <= i, num_pairs = n*(n+1)/2
        # row = floor((-1 + sqrt(1 + 8k)) / 2)
        sqrt_arg = (1 + 8 * idx).to(tl.float32)
        row_f = tl.math.floor((-1.0 + tl.math.sqrt(sqrt_arg)) * 0.5)
        i = row_f.to(tl.int32)
        # Handle floating-point edge case: if i*(i+1)/2 > idx, decrement i
        i = tl.where(i * (i + 1) > 2 * idx, i - 1, i)
        # col = k - row*(row+1)/2
        j = idx - (i * (i + 1)) // 2
    else:
        # Without self-loops: j < i, num_pairs = n*(n-1)/2
        # row = floor((1 + sqrt(1 + 8k)) / 2)
        sqrt_arg = (1 + 8 * idx).to(tl.float32)
        row_f = tl.math.floor((1.0 + tl.math.sqrt(sqrt_arg)) * 0.5)
        i = row_f.to(tl.int32)
        # Handle floating-point edge case (same correction as CUDA)
        i = tl.where(i * (i - 1) > 2 * idx, i - 1, i)
        # col = k - row*(row-1)/2
        j = idx - (i * (i - 1)) // 2

    # Validate indices: check bounds and triangular constraint
    # Due to float precision, we may get invalid (i, j) pairs
    valid = valid & (i >= 0) & (i < n_atoms) & (j >= 0) & (j <= i)
    if not loop:
        # For non-loop case, also require j < i (no self-loops)
        valid = valid & (j < i)

    batch_i = tl.load(batch_ptr + i, mask=valid, other=0)
    batch_j = tl.load(batch_ptr + j, mask=valid, other=0)
    valid = valid & (batch_i == batch_j)

    pos_ix = tl.load(pos_ptr + i * 3 + 0, mask=valid, other=0.0)
    pos_iy = tl.load(pos_ptr + i * 3 + 1, mask=valid, other=0.0)
    pos_iz = tl.load(pos_ptr + i * 3 + 2, mask=valid, other=0.0)
    pos_jx = tl.load(pos_ptr + j * 3 + 0, mask=valid, other=0.0)
    pos_jy = tl.load(pos_ptr + j * 3 + 1, mask=valid, other=0.0)
    pos_jz = tl.load(pos_ptr + j * 3 + 2, mask=valid, other=0.0)

    dx = pos_ix - pos_jx
    dy = pos_iy - pos_jy
    dz = pos_iz - pos_jz

    if use_periodic:
        box_base = box_ptr + batch_i * box_batch_stride

        b20 = tl.load(box_base + 2 * 3 + 0, mask=valid, other=0.0)
        b21 = tl.load(box_base + 2 * 3 + 1, mask=valid, other=0.0)
        b22 = tl.load(box_base + 2 * 3 + 2, mask=valid, other=1.0)
        b10 = tl.load(box_base + 1 * 3 + 0, mask=valid, other=0.0)
        b11 = tl.load(box_base + 1 * 3 + 1, mask=valid, other=1.0)
        b00 = tl.load(box_base + 0 * 3 + 0, mask=valid, other=1.0)

        scale3 = _tl_round(dz / b22)
        dx = dx - scale3 * b20
        dy = dy - scale3 * b21
        dz = dz - scale3 * b22
        scale2 = _tl_round(dy / b11)
        dx = dx - scale2 * b10
        dy = dy - scale2 * b11
        scale1 = _tl_round(dx / b00)
        dx = dx - scale1 * b00

    dist2 = dx * dx + dy * dy + dz * dz
    dist = tl.sqrt(dist2)
    # Self-loops (i == j) are exempt from cutoff_lower since they have distance 0
    is_self_loop = i == j
    valid = valid & (dist < cutoff_upper) & ((dist >= cutoff_lower) | is_self_loop)

    valid_int = valid.to(tl.int32)
    local_idx = tl.cumsum(valid_int, axis=0) - 1
    total_pairs = tl.sum(valid_int, axis=0)

    # For transpose, don't count self-loops (i == j)
    if include_transpose:
        is_not_self_loop = i != j
        valid_transpose = valid & is_not_self_loop
        total_transpose = tl.sum(valid_transpose.to(tl.int32), axis=0)
        total_out = total_pairs + total_transpose
    else:
        total_out = total_pairs

    has_work = total_out > 0
    start_idx = tl.atomic_add(counter_ptr, total_out, mask=has_work)
    start_idx = tl.where(has_work, start_idx, 0)
    start_idx = tl.broadcast_to(start_idx, local_idx.shape)

    write_idx = start_idx + local_idx
    mask_store = valid & (write_idx < max_pairs)
    tl.store(neighbors0_ptr + write_idx, i, mask=mask_store)
    tl.store(neighbors1_ptr + write_idx, j, mask=mask_store)
    tl.store(deltas_ptr + write_idx * 3 + 0, dx, mask=mask_store)
    tl.store(deltas_ptr + write_idx * 3 + 1, dy, mask=mask_store)
    tl.store(deltas_ptr + write_idx * 3 + 2, dz, mask=mask_store)
    tl.store(distances_ptr + write_idx, dist, mask=mask_store)

    if include_transpose:
        # Don't add transpose for self-loops (i == j)
        is_not_self_loop = i != j
        valid_transpose = valid & is_not_self_loop
        valid_t_int = valid_transpose.to(tl.int32)
        local_idx_t = tl.cumsum(valid_t_int, axis=0) - 1
        total_pairs_t = tl.sum(valid_t_int, axis=0)

        write_idx_t = start_idx + total_pairs + local_idx_t
        mask_store_t = valid_transpose & (write_idx_t < max_pairs)
        tl.store(neighbors0_ptr + write_idx_t, j, mask=mask_store_t)
        tl.store(neighbors1_ptr + write_idx_t, i, mask=mask_store_t)
        tl.store(deltas_ptr + write_idx_t * 3 + 0, -dx, mask=mask_store_t)
        tl.store(deltas_ptr + write_idx_t * 3 + 1, -dy, mask=mask_store_t)
        tl.store(deltas_ptr + write_idx_t * 3 + 2, -dz, mask=mask_store_t)
        tl.store(distances_ptr + write_idx_t, dist, mask=mask_store_t)


@triton_op("torchmdnet::triton_neighbor_bruteforce", mutates_args={})
def triton_neighbor_bruteforce(
    positions: Tensor,
    batch: Tensor,
    box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = positions.device
    dtype = positions.dtype
    n_atoms = positions.size(0)

    batch = batch.contiguous()
    positions = positions.contiguous()
    if use_periodic:
        if box_vectors.dim() == 2:
            box_vectors = box_vectors.unsqueeze(0)
        elif box_vectors.dim() != 3:
            raise ValueError('Expected "box_vectors" to have shape (n_batch, 3, 3)')
        box_vectors = box_vectors.to(device=device, dtype=dtype)
        box_vectors = box_vectors.contiguous()
        # Use stride 0 to broadcast single box to all batches (avoids CPU sync)
        box_batch_stride = 0 if box_vectors.size(0) == 1 else 9

    neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
    deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
    distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
    num_pairs = torch.zeros((1,), device=device, dtype=torch.int32)

    # Compute triangular pair count: n*(n-1)/2 without self-loops, n*(n+1)/2 with
    if loop:
        num_all_pairs = n_atoms * (n_atoms + 1) // 2
    else:
        num_all_pairs = n_atoms * (n_atoms - 1) // 2

    # Grid covers only triangular pairs (not n*n)
    grid = lambda meta: (triton.cdiv(num_all_pairs, meta["BLOCK"]),)

    wrap_triton(_neighbor_brute_kernel)[grid](
        positions,
        batch,
        box_vectors if use_periodic else positions,  # dummy pointer if not periodic
        neighbors[0],
        neighbors[1],
        deltas,
        distances,
        num_pairs,
        box_batch_stride if use_periodic else 0,
        n_atoms,
        num_all_pairs,
        use_periodic,
        include_transpose,
        loop,
        max_num_pairs,
        cutoff_lower,
        cutoff_upper,
        BLOCK=256,
    )

    return neighbors, deltas, distances, num_pairs


class TritonBruteNeighborAutograd(TritonNeighborAutograd):
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
        neighbors, deltas, distances, num_pairs = triton_neighbor_bruteforce(
            positions,
            batch,
            box_vectors,
            use_periodic,
            cutoff_lower,
            cutoff_upper,
            max_num_pairs,
            loop,
            include_transpose,
        )

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = positions.size(0)
        return neighbors, deltas, distances, num_pairs
