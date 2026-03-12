# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Tuple
import torch
from torch import Tensor
import triton
import triton.language as tl

from torchmdnet.extensions.neighbor_utils import (
    BaseNeighborAutograd as TritonNeighborAutograd,
    neighbor_grad_positions,
    neighbor_op_setup_context,
)


@triton.jit
def _tl_round(x):
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


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
            max_num_pairs,
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
            max_num_pairs,
            bool(loop),
            bool(include_transpose),
            num_cells,
        )
    else:
        raise ValueError(f"Unsupported strategy {strategy}")
