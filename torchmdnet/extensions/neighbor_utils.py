# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Shared utilities for neighbor list computation (backward, cell list building)."""

import torch
from torch import Tensor


def neighbor_grad_positions(ctx, grad_deltas, grad_distances):
    """Compute gradient of positions from neighbor list gradients.

    Shared backward logic for all neighbor list ops.
    Expects ctx to have saved_tensors = (neighbors, edge_vec, edge_weight)
    and ctx.num_atoms set.

    Returns grad_positions tensor of shape (num_atoms, 3).
    """
    neighbors, edge_vec, edge_weight = ctx.saved_tensors
    num_atoms = ctx.num_atoms

    if grad_deltas is None:
        grad_deltas = torch.zeros_like(edge_vec)
    if grad_distances is None:
        grad_distances = torch.zeros_like(edge_weight)

    zero_mask = edge_weight.eq(0)
    zero_mask3 = zero_mask.unsqueeze(-1).expand_as(grad_deltas)

    grad_distances_term = edge_vec / edge_weight.masked_fill(zero_mask, 1).unsqueeze(-1)
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

    return grad_positions


def neighbor_op_setup_context(ctx, inputs, output):
    """Shared setup_context for register_autograd on neighbor list ops."""
    positions = inputs[0]
    neighbors, deltas, distances, num_pairs = output
    ctx.save_for_backward(neighbors, deltas, distances)
    ctx.num_atoms = positions.size(0)


class BaseNeighborAutograd(torch.autograd.Function):
    """Base autograd function for neighbor list ops."""

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        grad_positions = neighbor_grad_positions(ctx, grad_deltas, grad_distances)
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


def get_cell_dimensions(
    box_x: torch.float32,
    box_y: torch.float32,
    box_z: torch.float32,
    cutoff_upper: torch.float32,
) -> int:
    """Compute cell grid dimensions from box sizes and cutoff."""
    nx = torch.floor(box_x / cutoff_upper).clamp(min=3).long()
    ny = torch.floor(box_y / cutoff_upper).clamp(min=3).long()
    nz = torch.floor(box_z / cutoff_upper).clamp(min=3).long()
    return torch.stack([nx, ny, nz])


def build_cell_list(
    positions: Tensor,
    batch: Tensor,
    box_sizes: Tensor,
    use_periodic: bool,
    cell_dims: Tensor,
    num_cells: int,
):
    """Build the cell list data structure using 1D sorted arrays.

    Args:
        positions: [N, 3] atom positions
        batch: [N] batch indices
        box_sizes: [3] box diagonal elements
        use_periodic: whether to use periodic boundary conditions
        cell_dims: [3] number of cells in each dimension (pre-computed)
        num_cells: total number of cells (pre-computed, fixed for CUDA graphs)

    Returns:
        sorted_indices: [n_atoms] original atom indices sorted by cell (int32)
        sorted_positions: [n_atoms, 3] positions sorted by cell
        sorted_batch: [n_atoms] batch indices sorted by cell
        cell_start: [num_cells] start index for each cell (int32)
        cell_end: [num_cells] end index (exclusive) for each cell (int32)
    """
    device = positions.device
    n_atoms = positions.size(0)

    if use_periodic:
        inv_box = 1.0 / box_sizes
        wrapped = positions - torch.floor(positions * inv_box) * box_sizes
    else:
        wrapped = positions + 0.5 * box_sizes

    cell_size = box_sizes / cell_dims.float()
    cell_coords = (wrapped / cell_size).long()
    cell_coords = torch.clamp(
        cell_coords, min=torch.zeros(3, device=device), max=cell_dims - 1
    )

    cell_idx = (
        cell_coords[:, 0] * (cell_dims[1] * cell_dims[2])
        + cell_coords[:, 1] * cell_dims[2]
        + cell_coords[:, 2]
    ).long()

    sorted_cell_idx, sort_order = torch.sort(cell_idx)
    sorted_indices = sort_order.int()

    sorted_positions = positions.index_select(0, sort_order).contiguous()
    sorted_batch = batch.index_select(0, sort_order).contiguous()

    cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_counts.scatter_add_(
        0, cell_idx, torch.ones(n_atoms, dtype=torch.int32, device=device)
    )

    cell_end = torch.cumsum(cell_counts, dim=0).int()
    cell_start = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_start[1:] = cell_end[:-1]

    return sorted_indices, sorted_positions, sorted_batch, cell_start, cell_end
