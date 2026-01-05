# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Tuple
import torch
from torch import Tensor
import triton
import triton.language as tl


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
    num_cells: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    from torchmdnet.extensions.triton_cell import TritonCellNeighborAutograd
    from torchmdnet.extensions.triton_brute import TritonBruteNeighborAutograd

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
            cutoff_lower,
            cutoff_upper,
            int(max_num_pairs),
            bool(loop),
            bool(include_transpose),
            num_cells,
        )
    else:
        raise ValueError(f"Unsupported strategy {strategy}")
