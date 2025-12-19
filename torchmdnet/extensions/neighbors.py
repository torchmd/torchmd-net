from torch import Tensor
from typing import Tuple
import torch


def _round_nearest(x: Tensor) -> Tensor:
    # Equivalent to torch.round but works for both float32/float64 and keeps TorchScript happy.
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _apply_pbc_torch(deltas: Tensor, box_for_pairs: Tensor) -> Tensor:
    # box_for_pairs: (num_pairs, 3, 3)
    scale3 = _round_nearest(deltas[:, 2] / box_for_pairs[:, 2, 2])
    deltas = deltas - scale3.unsqueeze(1) * box_for_pairs[:, 2]
    scale2 = _round_nearest(deltas[:, 1] / box_for_pairs[:, 1, 1])
    deltas = deltas - scale2.unsqueeze(1) * box_for_pairs[:, 1]
    scale1 = _round_nearest(deltas[:, 0] / box_for_pairs[:, 0, 0])
    deltas = deltas - scale1.unsqueeze(1) * box_for_pairs[:, 0]
    return deltas


def torch_neighbor_bruteforce(
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

    # Generate indices for all n*n pairs
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

    # Flatten everything: (n*n)
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
