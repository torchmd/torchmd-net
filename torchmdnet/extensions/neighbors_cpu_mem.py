# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch import Tensor
from typing import Tuple


def torch_neighbor_bruteforce_memory_efficient(
    strategy: str,
    positions: Tensor,
    batch: Tensor,
    in_box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute neighbor pairs within a cutoff distance using a more memory-efficient approach but not torch.compile compatible.

    Args:
        strategy: Strategy string (unused in CPU implementation)
        positions: Tensor of shape (n_atoms, 3) with atom positions
        batch: Tensor of shape (n_atoms,) with batch indices for each atom
        in_box_vectors: Box vectors for periodic boundary conditions, shape (3, 3) or (n_batch, 3, 3)
        use_periodic: Whether to use periodic boundary conditions
        cutoff_lower: Lower cutoff distance
        cutoff_upper: Upper cutoff distance
        max_num_pairs: Maximum number of pairs to return
        loop: Whether to include self-interactions (i, i)
        include_transpose: Whether to include both (i, j) and (j, i) pairs

    Returns:
        neighbors: Tensor of shape (2, num_pairs) with neighbor indices
        deltas: Tensor of shape (num_pairs, 3) with displacement vectors
        distances: Tensor of shape (num_pairs,) with distances
        num_pairs_found: Tensor of shape (1,) with the number of pairs found
    """
    from torchmdnet.extensions.neighbors import _round_nearest

    assert positions.dim() == 2, 'Expected "positions" to have two dimensions'
    assert (
        positions.size(0) > 0
    ), 'Expected the 1st dimension size of "positions" to be more than 0'
    assert (
        positions.size(1) == 3
    ), 'Expected the 2nd dimension size of "positions" to be 3'
    assert positions.is_contiguous(), 'Expected "positions" to be contiguous'
    assert cutoff_upper > 0, 'Expected "cutoff" to be positive'

    box_vectors = in_box_vectors
    n_batch = batch.max().item() + 1

    if use_periodic:
        if box_vectors.dim() == 2:
            box_vectors = box_vectors.unsqueeze(0).expand(n_batch, 3, 3)

        assert box_vectors.dim() == 3, 'Expected "box_vectors" to have three dimensions'
        assert (
            box_vectors.size(1) == 3 and box_vectors.size(2) == 3
        ), 'Expected "box_vectors" to have shape (n_batch, 3, 3)'
        assert (
            box_vectors.size(0) == n_batch
        ), 'Expected "box_vectors" to have shape (n_batch, 3, 3)'

        # Check that the box is a valid triclinic box, only check the first one
        v = box_vectors[0]
        c = cutoff_upper

        assert v[0, 1] == 0, "Invalid box vectors: box_vectors[0][1] != 0"
        assert v[0, 2] == 0, "Invalid box vectors: box_vectors[0][2] != 0"
        assert v[1, 2] == 0, "Invalid box vectors: box_vectors[1][2] != 0"
        assert v[0, 0] >= 2 * c, "Invalid box vectors: box_vectors[0][0] < 2*cutoff"
        assert v[1, 1] >= 2 * c, "Invalid box vectors: box_vectors[1][1] < 2*cutoff"
        assert v[2, 2] >= 2 * c, "Invalid box vectors: box_vectors[2][2] < 2*cutoff"
        assert (
            v[0, 0] >= 2 * v[1, 0]
        ), "Invalid box vectors: box_vectors[0][0] < 2*box_vectors[1][0]"
        assert (
            v[0, 0] >= 2 * v[2, 0]
        ), "Invalid box vectors: box_vectors[0][0] < 2*box_vectors[2][0]"
        assert (
            v[1, 1] >= 2 * v[2, 1]
        ), "Invalid box vectors: box_vectors[1][1] < 2*box_vectors[2][1]"

    assert max_num_pairs > 0, 'Expected "max_num_neighbors" to be positive'

    n_atoms = positions.size(0)

    # Generate all pairs using lower triangular indices
    neighbors = torch.tril_indices(
        n_atoms, n_atoms, offset=-1, device=positions.device, dtype=torch.int32
    )

    # Filter pairs to same batch
    mask = batch[neighbors[0]] == batch[neighbors[1]]
    neighbors = neighbors[:, mask].to(torch.int32)

    # Calculate deltas
    deltas = positions[neighbors[0]] - positions[neighbors[1]]

    if use_periodic:
        pair_batch = batch[neighbors[0]]

        # Apply periodic boundary conditions for triclinic box
        scale3 = _round_nearest(deltas[:, 2] / box_vectors[pair_batch, 2, 2])
        deltas[:, 0] = deltas[:, 0] - scale3 * box_vectors[pair_batch, 2, 0]
        deltas[:, 1] = deltas[:, 1] - scale3 * box_vectors[pair_batch, 2, 1]
        deltas[:, 2] = deltas[:, 2] - scale3 * box_vectors[pair_batch, 2, 2]

        scale2 = _round_nearest(deltas[:, 1] / box_vectors[pair_batch, 1, 1])
        deltas[:, 0] = deltas[:, 0] - scale2 * box_vectors[pair_batch, 1, 0]
        deltas[:, 1] = deltas[:, 1] - scale2 * box_vectors[pair_batch, 1, 1]

        scale1 = _round_nearest(deltas[:, 0] / box_vectors[pair_batch, 0, 0])
        deltas[:, 0] = deltas[:, 0] - scale1 * box_vectors[pair_batch, 0, 0]

    # Calculate distances (gradient-safe for zero distances)
    dist_sq = (deltas * deltas).sum(dim=-1)
    zero_mask = dist_sq == 0
    distances = torch.where(
        zero_mask,
        torch.zeros_like(dist_sq),
        torch.sqrt(dist_sq.clamp(min=1e-32)),
    )

    # Filter by cutoff
    mask = (distances < cutoff_upper) & (distances >= cutoff_lower)
    neighbors = neighbors[:, mask]
    deltas = deltas[mask]
    distances = distances[mask]

    if include_transpose:
        neighbors = torch.hstack([neighbors, torch.stack([neighbors[1], neighbors[0]])])
        distances = torch.hstack([distances, distances])
        deltas = torch.vstack([deltas, -deltas])

    if loop:
        range_tensor = torch.arange(
            0, n_atoms, dtype=torch.int32, device=positions.device
        )
        neighbors = torch.hstack([neighbors, torch.stack([range_tensor, range_tensor])])
        distances = torch.hstack(
            [distances, torch.zeros_like(range_tensor, dtype=distances.dtype)]
        )
        deltas = torch.vstack(
            [deltas, torch.zeros(n_atoms, 3, dtype=deltas.dtype, device=deltas.device)]
        )

    num_pairs_found = torch.tensor(
        [distances.size(0)], dtype=torch.int32, device=positions.device
    )

    # Pad to max_num_pairs to enable torch.compile by guaranteeing predictable output size
    extension = max(max_num_pairs - distances.size(0), 0)
    if extension > 0:
        deltas = torch.vstack(
            [
                deltas,
                torch.zeros(extension, 3, dtype=deltas.dtype, device=deltas.device),
            ]
        )
        distances = torch.hstack(
            [
                distances,
                torch.zeros(extension, dtype=distances.dtype, device=distances.device),
            ]
        )
        # For the neighbors add (-1, -1) pairs to fill the tensor
        neighbors = torch.hstack(
            [
                neighbors,
                torch.full(
                    (2, extension), -1, dtype=torch.int32, device=neighbors.device
                ),
            ]
        )

    return neighbors, deltas, distances, num_pairs_found
