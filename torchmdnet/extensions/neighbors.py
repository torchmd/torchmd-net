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
    """Optimized brute-force neighbor list using pure PyTorch.

    This implementation avoids nonzero() to be torch.compile compatible.
    Uses triangular indexing to reduce memory usage and computation from O(n^2) to O(n^2/2).
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

    # Generate base pairs
    if loop:
        # loop=True: i >= j (lower triangle including diagonal)
        tril_indices = torch.tril_indices(n_atoms, n_atoms, device=device)
        i_indices = tril_indices[0]
        j_indices = tril_indices[1]
    else:
        # loop=False: i > j (lower triangle excluding diagonal)
        tril_indices = torch.tril_indices(n_atoms, n_atoms, offset=-1, device=device)
        i_indices = tril_indices[0]
        j_indices = tril_indices[1]

    # If include_transpose, add the flipped pairs (j,i)
    if include_transpose:
        if loop:
            # For loop=True, base pairs are i >= j, so add i < j transposes
            triu_indices = torch.triu_indices(n_atoms, n_atoms, offset=1, device=device)
            i_transpose = triu_indices[0]
            j_transpose = triu_indices[1]
        else:
            # For loop=False, base pairs are i > j, so add all transposes (j,i)
            i_transpose = j_indices
            j_transpose = i_indices
        # Combine base and transpose pairs
        i_indices = torch.cat([i_indices, i_transpose])
        j_indices = torch.cat([j_indices, j_transpose])

    # Compute deltas for all pairs
    deltas = positions[i_indices] - positions[j_indices]

    # Apply PBC if needed
    if use_periodic:
        batch_i = batch[i_indices]
        if box_vectors.size(0) == 1:
            # Single box for all - use the same box
            box_for_pairs = box_vectors.expand(len(deltas), 3, 3)
        else:
            # Per-batch boxes - index by batch of atom i
            box_for_pairs = box_vectors[batch_i]
        # Apply PBC to pairs
        deltas = _apply_pbc_torch(deltas, box_for_pairs)

    # Compute distances for all pairs
    dist_sq = (deltas * deltas).sum(dim=-1)
    zero_mask = dist_sq == 0
    distances = torch.where(
        zero_mask,
        torch.zeros_like(dist_sq),
        torch.sqrt(dist_sq.clamp(min=1e-32)),
    )

    # Build validity mask for all pairs
    valid_mask = torch.ones(len(distances), device=device, dtype=torch.bool)

    # Apply batch constraint
    if batch.numel() > 0:
        same_batch = batch[i_indices] == batch[j_indices]
        valid_mask = valid_mask & same_batch

    # Apply cutoff constraints
    # Self-loops (i == j) are exempt from cutoff_lower since they have distance 0
    is_self_loop = i_indices == j_indices
    valid_mask = (
        valid_mask
        & (distances < cutoff_upper)
        & ((distances >= cutoff_lower) | is_self_loop)
    )

    # Sort key: valid pairs by distance, invalid pairs get infinity (sorted last)
    sort_key = torch.where(
        valid_mask,
        distances,
        torch.full_like(distances, float("inf")),
    )

    # Sort and take top max_num_pairs (fixed output shape)
    # For include_transpose + loop case, we may have more pairs than expected
    num_candidates = min(sort_key.size(0), max_num_pairs)
    order = torch.argsort(sort_key)[:num_candidates]

    # Gather results using the sorted indices
    i_out = i_indices.index_select(0, order)
    j_out = j_indices.index_select(0, order)
    deltas_out = deltas.index_select(0, order)
    distances_out = distances.index_select(0, order)
    valid_out = valid_mask.index_select(0, order)

    # Pad to max_num_pairs if needed
    if num_candidates < max_num_pairs:
        pad_size = max_num_pairs - num_candidates
        pad_indices = torch.full((pad_size,), -1, device=device, dtype=torch.long)
        pad_deltas = torch.zeros((pad_size, 3), device=device, dtype=dtype)
        pad_distances = torch.zeros((pad_size,), device=device, dtype=dtype)
        pad_valid = torch.zeros((pad_size,), device=device, dtype=torch.bool)

        i_out = torch.cat([i_out, pad_indices])
        j_out = torch.cat([j_out, pad_indices])
        deltas_out = torch.cat([deltas_out, pad_deltas])
        distances_out = torch.cat([distances_out, pad_distances])
        valid_out = torch.cat([valid_out, pad_valid])

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

    # Count valid pairs (before slicing) to detect overflow
    num_pairs_tensor = valid_mask.sum().view(1).to(torch.long)
    return neighbors_out, deltas_out, distances_out, num_pairs_tensor
