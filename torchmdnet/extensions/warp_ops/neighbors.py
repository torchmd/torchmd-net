"""Warp-based neighbor list ops with PyTorch custom op integration.

Provides brute-force and cell-list neighbor pair computation using NVIDIA Warp kernels,
with torch.library.custom_op registration, fake/meta implementations for torch.compile,
and autograd support via the shared PyTorch backward in neighbor_utils.
"""

from __future__ import annotations

from typing import Tuple

import torch
import warp as wp
from torch import Tensor

from torchmdnet.extensions.neighbor_utils import (
    BaseNeighborAutograd,
    build_cell_list,
    get_cell_dimensions,
    neighbor_grad_positions,
    neighbor_op_setup_context,
)
from torchmdnet.extensions.warp_kernels import get_module, get_stream
from torchmdnet.extensions.warp_kernels.neighbors_brute import (
    get_brute_specialized_kernel,
)


# ---------------------------------------------------------------------------
# Brute-force kernel op
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "torchmdnet::warp_neighbor_brute_fwd",
    mutates_args=(),
    device_types=["cuda", "cpu"],
)
def warp_neighbor_brute_fwd(
    positions: Tensor,
    batch: Tensor,
    box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> list[Tensor]:
    device = positions.device
    dtype = positions.dtype
    n_atoms = positions.size(0)

    positions = positions.contiguous()
    batch = batch.contiguous()

    if use_periodic:
        if box_vectors.dim() == 2:
            box_vectors = box_vectors.unsqueeze(0)
        elif box_vectors.dim() != 3:
            raise ValueError('Expected "box_vectors" to have shape (n_batch, 3, 3)')
        box_vectors = box_vectors.to(device=device, dtype=dtype).contiguous()
        box_flat = box_vectors.reshape(-1).contiguous()
        box_batch_stride = 0 if box_vectors.size(0) == 1 else 9
    else:
        box_flat = torch.zeros(9, device=device, dtype=dtype)
        box_batch_stride = 0

    neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
    deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
    distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
    num_pairs = torch.zeros((1,), device=device, dtype=torch.int32)

    if loop:
        num_all_pairs = n_atoms * (n_atoms + 1) // 2
    else:
        num_all_pairs = n_atoms * (n_atoms - 1) // 2

    if num_all_pairs == 0:
        return [neighbors, deltas, distances, num_pairs]

    stream = get_stream(device)
    wp_device = wp.device_from_torch(device)

    pos_wp = wp.from_torch(positions.detach(), return_ctype=True)
    batch_wp = wp.from_torch(batch.detach(), return_ctype=True)
    box_wp = wp.from_torch(box_flat.detach(), return_ctype=True)
    neighbors_wp = wp.from_torch(neighbors.detach(), return_ctype=True)
    deltas_wp = wp.from_torch(deltas.detach(), return_ctype=True)
    distances_wp = wp.from_torch(distances.detach(), return_ctype=True)
    counter_wp = wp.from_torch(num_pairs.detach(), return_ctype=True)

    if num_all_pairs <= 2**31 - 1:
        kernel = get_brute_specialized_kernel(
            str(dtype), use_periodic, include_transpose, loop
        )
        wp.launch(
            kernel,
            dim=(num_all_pairs,),
            stream=stream,
            device=wp_device,
            inputs=(
                pos_wp,
                batch_wp,
                box_wp,
                neighbors_wp,
                deltas_wp,
                distances_wp,
                counter_wp,
                n_atoms,
                num_all_pairs,
                max_num_pairs,
                box_batch_stride,
                cutoff_lower**2,
                cutoff_upper**2,
            ),
        )
    else:
        _GRID_DIM_Y = 32768
        grid_y = min(num_all_pairs, _GRID_DIM_Y)
        grid_x = (num_all_pairs + grid_y - 1) // grid_y
        kernel = get_module("neighbors_brute_fwd_i64", [str(dtype)])
        wp.launch(
            kernel,
            dim=(grid_x, grid_y),
            stream=stream,
            device=wp_device,
            inputs=(
                pos_wp,
                batch_wp,
                box_wp,
                neighbors_wp,
                deltas_wp,
                distances_wp,
                counter_wp,
                n_atoms,
                max_num_pairs,
                box_batch_stride,
                1 if use_periodic else 0,
                1 if include_transpose else 0,
                1 if loop else 0,
                cutoff_lower**2,
                cutoff_upper**2,
                grid_y,
            ),
        )

    return [neighbors, deltas, distances, num_pairs]


@torch.library.register_fake("torchmdnet::warp_neighbor_brute_fwd")
def _(
    positions: Tensor,
    batch: Tensor,
    box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> list[Tensor]:
    return [
        torch.empty((2, max_num_pairs), device=positions.device, dtype=torch.long),
        torch.empty((max_num_pairs, 3), device=positions.device, dtype=positions.dtype),
        torch.empty((max_num_pairs,), device=positions.device, dtype=positions.dtype),
        torch.empty((1,), device=positions.device, dtype=torch.int32),
    ]


def _brute_setup_context(ctx, inputs, output):
    positions = inputs[0]
    neighbors, deltas, distances, num_pairs = output
    ctx.save_for_backward(neighbors, deltas, distances)
    ctx.num_atoms = positions.size(0)


def _brute_backward(ctx, grads):
    _, grad_deltas, grad_distances, _ = grads
    grad_positions = neighbor_grad_positions(ctx, grad_deltas, grad_distances)
    return grad_positions, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "torchmdnet::warp_neighbor_brute_fwd",
    _brute_backward,
    setup_context=_brute_setup_context,
)


# ---------------------------------------------------------------------------
# Cell-list kernel op
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "torchmdnet::warp_neighbor_cell_fwd",
    mutates_args=(),
    device_types=["cuda", "cpu"],
)
def warp_neighbor_cell_fwd(
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
) -> list[Tensor]:
    device = positions.device
    dtype = positions.dtype
    n_atoms = positions.size(0)

    if positions.dim() != 2 or positions.size(1) != 3:
        raise ValueError('Expected "positions" to have shape (N, 3)')
    if batch.dim() != 1 or batch.size(0) != n_atoms:
        raise ValueError('Expected "batch" to have shape (N,)')

    box_vectors = box_vectors.contiguous()
    if box_vectors.dim() == 3:
        box_diag = box_vectors[0]
    else:
        box_diag = box_vectors
    box_sizes = torch.stack(
        [box_diag[0, 0], box_diag[1, 1], box_diag[2, 2]]
    ).contiguous()

    cell_dims = get_cell_dimensions(
        box_sizes[0], box_sizes[1], box_sizes[2], cutoff_upper
    )

    sorted_indices, sorted_positions, sorted_batch, cell_start, cell_end = (
        build_cell_list(positions, batch, box_sizes, use_periodic, cell_dims, num_cells)
    )

    neighbors = torch.full((2, max_num_pairs), -1, device=device, dtype=torch.long)
    deltas = torch.zeros((max_num_pairs, 3), device=device, dtype=dtype)
    distances = torch.zeros((max_num_pairs,), device=device, dtype=dtype)
    num_pairs = torch.zeros((1,), device=device, dtype=torch.int32)

    cell_dims_i32 = cell_dims.to(dtype=torch.int32).contiguous()

    stream = get_stream(device)
    wp_device = wp.device_from_torch(device)

    spos_wp = wp.from_torch(sorted_positions.detach(), return_ctype=True)
    sidx_wp = wp.from_torch(sorted_indices.detach(), return_ctype=True)
    sbatch_wp = wp.from_torch(sorted_batch.detach(), return_ctype=True)
    cstart_wp = wp.from_torch(cell_start.detach(), return_ctype=True)
    cend_wp = wp.from_torch(cell_end.detach(), return_ctype=True)
    bsizes_wp = wp.from_torch(box_sizes.detach(), return_ctype=True)
    cdims_wp = wp.from_torch(cell_dims_i32.detach(), return_ctype=True)
    neighbors_wp = wp.from_torch(neighbors.detach(), return_ctype=True)
    deltas_wp = wp.from_torch(deltas.detach(), return_ctype=True)
    distances_wp = wp.from_torch(distances.detach(), return_ctype=True)
    counter_wp = wp.from_torch(num_pairs.detach(), return_ctype=True)

    kernel = get_module("neighbors_cell_fwd", [str(dtype)])
    wp.launch(
        kernel,
        dim=(n_atoms,),
        stream=stream,
        device=wp_device,
        inputs=(
            spos_wp,
            sidx_wp,
            sbatch_wp,
            cstart_wp,
            cend_wp,
            bsizes_wp,
            cdims_wp,
            neighbors_wp,
            deltas_wp,
            distances_wp,
            counter_wp,
            n_atoms,
            max_num_pairs,
            1 if use_periodic else 0,
            1 if include_transpose else 0,
            1 if loop else 0,
            cutoff_lower**2,
            cutoff_upper**2,
        ),
    )

    return [neighbors, deltas, distances, num_pairs]


@torch.library.register_fake("torchmdnet::warp_neighbor_cell_fwd")
def _(
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
) -> list[Tensor]:
    return [
        torch.empty((2, max_num_pairs), device=positions.device, dtype=torch.long),
        torch.empty((max_num_pairs, 3), device=positions.device, dtype=positions.dtype),
        torch.empty((max_num_pairs,), device=positions.device, dtype=positions.dtype),
        torch.empty((1,), device=positions.device, dtype=torch.int32),
    ]


def _cell_setup_context(ctx, inputs, output):
    positions = inputs[0]
    neighbors, deltas, distances, num_pairs = output
    ctx.save_for_backward(neighbors, deltas, distances)
    ctx.num_atoms = positions.size(0)


def _cell_backward(ctx, grads):
    _, grad_deltas, grad_distances, _ = grads
    grad_positions = neighbor_grad_positions(ctx, grad_deltas, grad_distances)
    return grad_positions, None, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "torchmdnet::warp_neighbor_cell_fwd",
    _cell_backward,
    setup_context=_cell_setup_context,
)


# ---------------------------------------------------------------------------
# Autograd Function wrappers (used by the dispatch layer)
# ---------------------------------------------------------------------------


class WarpBruteNeighborAutograd(BaseNeighborAutograd):
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
        neighbors, deltas, distances, num_pairs = warp_neighbor_brute_fwd(
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


class WarpCellNeighborAutograd(BaseNeighborAutograd):
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
        num_cells: int,
    ):
        neighbors, deltas, distances, num_pairs = warp_neighbor_cell_fwd(
            positions,
            batch,
            box_vectors,
            use_periodic,
            cutoff_lower,
            cutoff_upper,
            max_num_pairs,
            loop,
            include_transpose,
            num_cells,
        )
        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = positions.size(0)
        return neighbors, deltas, distances, num_pairs

    @staticmethod
    def backward(ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs):  # type: ignore[override]
        parent_grads = BaseNeighborAutograd.backward(
            ctx, grad_neighbors, grad_deltas, grad_distances, grad_num_pairs
        )
        return (*parent_grads, None)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def warp_neighbor_pairs(
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
    """Dispatch to the appropriate Warp neighbor list kernel."""
    if positions.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("Unsupported dtype for Warp neighbor list")

    if strategy == "brute":
        return WarpBruteNeighborAutograd.apply(
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
        return WarpCellNeighborAutograd.apply(
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
