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


def _get_cell_dimensions(
    box_vectors: Tensor, cutoff_upper: float
) -> Tuple[int, int, int]:
    """Return (nx, ny, nz) cell counts following the CUDA cell list logic."""
    # box_vectors is either (3, 3) or (1, 3, 3) and already validated as diagonal
    if box_vectors.dim() == 3:
        box_diag = box_vectors[0]
    else:
        box_diag = box_vectors
    nx = int(max(math.floor(float(box_diag[0, 0]) / cutoff_upper), 3))
    ny = int(max(math.floor(float(box_diag[1, 1]) / cutoff_upper), 3))
    nz = int(max(math.floor(float(box_diag[2, 2]) / cutoff_upper), 3))
    if nx > 1024 or ny > 1024 or nz > 1024:
        raise RuntimeError("Too many cells in one dimension. Maximum is 1024")
    return nx, ny, nz


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


class TritonNeighborAutograd(torch.autograd.Function):
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
    from torchmdnet.extensions.triton_cell import TritonCellNeighborAutograd
    from torchmdnet.extensions.triton_brute import TritonBruteNeighborAutograd

    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if positions.device.type != "cuda":
        raise RuntimeError("Triton neighbor list requires CUDA tensors")
    if positions.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("Unsupported dtype for Triton neighbor list")

    if strategy == "brute":
        return TritonBruteNeighborAutograd.apply(
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
    elif strategy == "cell":
        return TritonCellNeighborAutograd.apply(
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
    elif strategy == "shared":
        # return TritonSharedNeighborAutograd.apply(
        #     positions,
        #     batch,
        #     box_vectors,
        #     use_periodic,
        #     float(cutoff_lower),
        #     float(cutoff_upper),
        #     int(max_num_pairs),
        # )
        pass
    else:
        raise ValueError(f"Unsupported strategy {strategy}")


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
    # Strategy is ignored for torch version. We only need the brute-force fallback
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
