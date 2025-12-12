import math
from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _round_nearest(x: Tensor) -> Tensor:
    # Equivalent to torch.round but works for both float32/float64 and keeps TorchScript happy.
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _validate_box(box_vectors: Tensor, cutoff_upper: float, n_batch: int) -> Tensor:
    if box_vectors.dim() == 2:
        box_vectors = box_vectors.unsqueeze(0).expand(n_batch, 3, 3)
    if box_vectors.dim() != 3 or box_vectors.shape[1:] != (3, 3):
        raise ValueError('Expected "box_vectors" to have shape (n_batch, 3, 3)')
    if box_vectors.size(0) != n_batch:
        raise ValueError('Expected "box_vectors" first dimension to match batch size')
    v = box_vectors[0]
    c = float(cutoff_upper)
    if not (
        v[0, 1] == 0
        and v[0, 2] == 0
        and v[1, 2] == 0
        and v[0, 0] >= 2 * c
        and v[1, 1] >= 2 * c
        and v[2, 2] >= 2 * c
        and v[0, 0] >= 2 * v[1, 0]
        and v[0, 0] >= 2 * v[2, 0]
        and v[1, 1] >= 2 * v[2, 1]
    ):
        raise ValueError("Invalid box vectors")
    return box_vectors


def _apply_pbc_torch(deltas: Tensor, box_for_pairs: Tensor) -> Tensor:
    # box_for_pairs: (num_pairs, 3, 3)
    scale3 = _round_nearest(deltas[:, 2] / box_for_pairs[:, 2, 2])
    deltas = deltas - scale3.unsqueeze(1) * box_for_pairs[:, 2]
    scale2 = _round_nearest(deltas[:, 1] / box_for_pairs[:, 1, 1])
    deltas = deltas - scale2.unsqueeze(1) * box_for_pairs[:, 1]
    scale1 = _round_nearest(deltas[:, 0] / box_for_pairs[:, 0, 0])
    deltas = deltas - scale1.unsqueeze(1) * box_for_pairs[:, 0]
    return deltas


def _torch_neighbor_bruteforce(
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
    """Brute-force neighbor list using pure PyTorch.

    This implementation avoids nonzero() to be torch.compile compatible.
    It uses argsort with infinity for invalid pairs to achieve fixed output shapes.
    """
    if positions.dim() != 2 or positions.size(1) != 3:
        raise ValueError('Expected "positions" to have shape (N, 3)')
    if batch.dim() != 1 or batch.size(0) != positions.size(0):
        raise ValueError('Expected "batch" to have shape (N,)')
    if max_num_pairs <= 0:
        raise ValueError('Expected "max_num_pairs" to be positive')

    device = positions.device
    dtype = positions.dtype
    n_atoms = positions.size(0)

    if use_periodic:
        if box_vectors.dim() == 2:
            box_vectors = box_vectors.unsqueeze(0)
        elif box_vectors.dim() != 3:
            raise ValueError('Expected "box_vectors" to have shape (n_batch, 3, 3)')
        box_vectors = box_vectors.to(device=device, dtype=dtype)

    # Generate indices for all n² pairs
    arange_n = torch.arange(n_atoms, device=device)
    i_idx = arange_n.view(-1, 1).expand(n_atoms, n_atoms)  # (n, n)
    j_idx = arange_n.view(1, -1).expand(n_atoms, n_atoms)  # (n, n)

    # Compute deltas for ALL pairs: (n, n, 3)
    pos_i = positions.unsqueeze(1)  # (n, 1, 3)
    pos_j = positions.unsqueeze(0)  # (1, n, 3)
    deltas_grid = pos_i - pos_j  # (n, n, 3)

    # Apply PBC if needed
    if use_periodic:
        # Get batch indices for all pairs
        batch_i = batch.view(-1, 1).expand(n_atoms, n_atoms)  # (n, n)
        if box_vectors.size(0) == 1:
            # Single box for all - broadcast
            box_for_grid = box_vectors.expand(n_atoms, n_atoms, 3, 3)  # (n, n, 3, 3)
        else:
            # Per-batch boxes - index by batch of atom i
            box_for_grid = box_vectors[batch_i]  # (n, n, 3, 3)
        # Apply PBC: deltas_grid is (n, n, 3), box_for_grid is (n, n, 3, 3)
        deltas_flat_pbc = deltas_grid.reshape(-1, 3)
        box_flat_pbc = box_for_grid.reshape(-1, 3, 3)
        deltas_flat_pbc = _apply_pbc_torch(deltas_flat_pbc, box_flat_pbc)
        deltas_grid = deltas_flat_pbc.reshape(n_atoms, n_atoms, 3)

    # Compute distances for all pairs: (n, n)
    dist_sq = (deltas_grid * deltas_grid).sum(dim=-1)
    zero_mask = dist_sq == 0
    distances_grid = torch.where(
        zero_mask,
        torch.zeros_like(dist_sq),
        torch.sqrt(dist_sq.clamp(min=1e-32)),
    )

    # Build validity mask (n, n)
    if include_transpose:
        valid_mask = torch.ones((n_atoms, n_atoms), device=device, dtype=torch.bool)
        if not loop:
            valid_mask = valid_mask & (i_idx != j_idx)
    else:
        valid_mask = (i_idx > j_idx) if not loop else (i_idx >= j_idx)

    # Apply batch constraint
    if batch.numel() > 0:
        same_batch = batch.view(-1, 1) == batch.view(1, -1)
        valid_mask = valid_mask & same_batch

    # Apply cutoff constraints
    # Self-loops (i == j) are exempt from cutoff_lower since they have distance 0
    is_self_loop = i_idx == j_idx
    valid_mask = (
        valid_mask
        & (distances_grid < cutoff_upper)
        & ((distances_grid >= cutoff_lower) | is_self_loop)
    )

    # Flatten everything: (n²)
    i_flat = i_idx.reshape(-1)
    j_flat = j_idx.reshape(-1)
    deltas_flat = deltas_grid.reshape(-1, 3)
    distances_flat = distances_grid.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Sort key: valid pairs by distance, invalid pairs get infinity (sorted last)
    sort_key = torch.where(
        valid_flat,
        distances_flat,
        torch.full_like(distances_flat, float("inf")),
    )

    # Sort and take top max_num_pairs (fixed output shape)
    order = torch.argsort(sort_key)[:max_num_pairs]

    # Gather results using the sorted indices
    i_out = i_flat.index_select(0, order)
    j_out = j_flat.index_select(0, order)
    deltas_out = deltas_flat.index_select(0, order)
    distances_out = distances_flat.index_select(0, order)
    valid_out = valid_flat.index_select(0, order)

    # Replace invalid entries with -1 (neighbors) and 0 (deltas/distances)
    # Ensure gradients flow through valid entries
    neighbors_out = torch.stack(
        [
            torch.where(valid_out, i_out, torch.full_like(i_out, -1)),
            torch.where(valid_out, j_out, torch.full_like(j_out, -1)),
        ]
    )
    # For deltas/distances, use where to preserve gradients for valid entries
    zero_deltas = deltas_out.detach() * 0
    zero_distances = distances_out.detach() * 0
    deltas_out = torch.where(valid_out.unsqueeze(1), deltas_out, zero_deltas)
    distances_out = torch.where(valid_out, distances_out, zero_distances)

    # Count ALL valid pairs (before slicing) to detect overflow
    num_pairs_tensor = valid_flat.sum().view(1).to(torch.long)
    return neighbors_out, deltas_out, distances_out, num_pairs_tensor


@triton.jit
def _tl_round(x):
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


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
    stride_pos_0: tl.constexpr,
    stride_pos_1: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_box_b: tl.constexpr,
    stride_box_r: tl.constexpr,
    stride_box_c: tl.constexpr,
    n_atoms: tl.constexpr,
    num_all_pairs: tl.constexpr,
    use_periodic: tl.constexpr,
    include_transpose: tl.constexpr,
    loop: tl.constexpr,
    max_pairs: tl.constexpr,
    cutoff_lower: tl.constexpr,
    cutoff_upper: tl.constexpr,
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

    batch_i = tl.load(batch_ptr + i * stride_batch, mask=valid, other=0)
    batch_j = tl.load(batch_ptr + j * stride_batch, mask=valid, other=0)
    valid = valid & (batch_i == batch_j)

    pos_ix = tl.load(
        pos_ptr + i * stride_pos_0 + 0 * stride_pos_1, mask=valid, other=0.0
    )
    pos_iy = tl.load(
        pos_ptr + i * stride_pos_0 + 1 * stride_pos_1, mask=valid, other=0.0
    )
    pos_iz = tl.load(
        pos_ptr + i * stride_pos_0 + 2 * stride_pos_1, mask=valid, other=0.0
    )
    pos_jx = tl.load(
        pos_ptr + j * stride_pos_0 + 0 * stride_pos_1, mask=valid, other=0.0
    )
    pos_jy = tl.load(
        pos_ptr + j * stride_pos_0 + 1 * stride_pos_1, mask=valid, other=0.0
    )
    pos_jz = tl.load(
        pos_ptr + j * stride_pos_0 + 2 * stride_pos_1, mask=valid, other=0.0
    )

    dx = pos_ix - pos_jx
    dy = pos_iy - pos_jy
    dz = pos_iz - pos_jz

    if use_periodic:
        box_base = box_ptr + batch_i * stride_box_b

        b20 = tl.load(
            box_base + 2 * stride_box_r + 0 * stride_box_c, mask=valid, other=0.0
        )
        b21 = tl.load(
            box_base + 2 * stride_box_r + 1 * stride_box_c, mask=valid, other=0.0
        )
        b22 = tl.load(
            box_base + 2 * stride_box_r + 2 * stride_box_c, mask=valid, other=1.0
        )
        b10 = tl.load(
            box_base + 1 * stride_box_r + 0 * stride_box_c, mask=valid, other=0.0
        )
        b11 = tl.load(
            box_base + 1 * stride_box_r + 1 * stride_box_c, mask=valid, other=1.0
        )
        b00 = tl.load(
            box_base + 0 * stride_box_r + 0 * stride_box_c, mask=valid, other=1.0
        )

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


class TritonNeighborAutograd(torch.autograd.Function):
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
            box_stride_0 = 0 if box_vectors.size(0) == 1 else box_vectors.stride(0)

        neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
        deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
        distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
        counter = torch.zeros((1,), device=device, dtype=torch.int32)

        # Compute triangular pair count: n*(n-1)/2 without self-loops, n*(n+1)/2 with
        if loop:
            num_all_pairs = n_atoms * (n_atoms + 1) // 2
        else:
            num_all_pairs = n_atoms * (n_atoms - 1) // 2

        # Grid covers only triangular pairs (not n²)
        grid = lambda meta: (triton.cdiv(num_all_pairs, meta["BLOCK"]),)

        _neighbor_brute_kernel[grid](
            positions,
            batch,
            box_vectors if use_periodic else positions,  # dummy pointer if not periodic
            neighbors[0],
            neighbors[1],
            deltas,
            distances,
            counter,
            positions.stride(0),
            positions.stride(1),
            batch.stride(0),
            box_stride_0 if use_periodic else 0,
            box_vectors.stride(1) if use_periodic else 0,
            box_vectors.stride(2) if use_periodic else 0,
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

        num_pairs = counter.to(torch.long)

        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = n_atoms
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        neighbors, edge_vec, edge_weight = ctx.saved_tensors
        num_atoms = ctx.num_atoms

        if grad_deltas is None:
            grad_deltas = torch.zeros_like(edge_vec)
        if grad_distances is None:
            grad_distances = torch.zeros_like(edge_weight)

        zero_mask = edge_weight.eq(0)
        zero_mask3 = zero_mask.unsqueeze(-1).expand_as(grad_deltas)

        grad_distances_term = edge_vec / edge_weight.masked_fill(
            zero_mask, 1
        ).unsqueeze(-1)
        grad_distances_term = grad_distances_term * grad_distances.masked_fill(
            zero_mask, 0
        ).unsqueeze(-1)

        grad_positions = torch.zeros(
            (num_atoms, 3), device=edge_vec.device, dtype=edge_vec.dtype
        )
        edge_index_safe = neighbors.masked_fill(
            zero_mask.unsqueeze(0).expand_as(neighbors), 0
        )
        grad_vec = grad_deltas.masked_fill(zero_mask3, 0) + grad_distances_term
        grad_positions.index_add_(0, edge_index_safe[0], grad_vec)
        grad_positions.index_add_(0, edge_index_safe[1], -grad_vec)

        return (
            grad_positions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_neighbor_pairs(
    strategy: str,
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
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if positions.device.type != "cuda":
        raise RuntimeError("Triton neighbor list requires CUDA tensors")
    if positions.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("Unsupported dtype for Triton neighbor list")

    return TritonNeighborAutograd.apply(
        positions,
        batch,
        box_vectors,
        use_periodic,
        float(cutoff_lower),
        float(cutoff_upper),
        int(max_num_pairs),
        bool(loop),
        bool(include_transpose),
    )


def torch_neighbor_pairs(
    strategy: str,
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
    return _torch_neighbor_bruteforce(
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
