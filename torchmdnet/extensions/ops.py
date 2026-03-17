# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch import Tensor
from typing import Tuple

from torchmdnet.extensions.warp_ops.neighbors import warp_neighbor_pairs

__all__ = ["get_neighbor_pairs_kernel"]


def get_neighbor_pairs_kernel(
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
    """Computes the neighbor pairs for a given set of atomic positions.
    The list is generated as a list of pairs (i,j) without any enforced ordering.
    The list is padded with -1 to the maximum number of pairs.

    Parameters
    ----------
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["brute", "cell"]`.
    positions : Tensor
        A tensor with shape (N, 3) representing the atomic positions.
    batch : Tensor
        A tensor with shape (N,). Specifies the batch for each atom.
    box_vectors : Tensor
        The vectors defining the periodic box with shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a different box is used for each sample.
    use_periodic : bool
        Whether to apply periodic boundary conditions.
    cutoff_lower : float
        Lower cutoff for the neighbor list.
    cutoff_upper : float
        Upper cutoff for the neighbor list.
    max_num_pairs : int
        Maximum number of pairs to store.
    loop : bool
        Whether to include self-interactions.
    include_transpose : bool
        Whether to include the transpose of the neighbor list (pair i,j and pair j,i).
    num_cells : int
        The number of cells in the grid if using the cell strategy.
    Returns
    -------
    neighbors : Tensor
        List of neighbors for each atom. Shape (2, max_num_pairs).
    distances : Tensor
        List of distances for each atom. Shape (max_num_pairs,).
    distance_vecs : Tensor
        List of distance vectors for each atom. Shape (max_num_pairs, 3).
    num_pairs : Tensor
        The number of pairs found.
    """
    if torch.jit.is_scripting():
        if strategy == "brute":
            result = torch.ops.torchmdnet.warp_neighbor_brute_fwd(
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
        else:
            result = torch.ops.torchmdnet.warp_neighbor_cell_fwd(
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
        return result[0], result[1], result[2], result[3]

    return warp_neighbor_pairs(
        strategy=strategy,
        positions=positions,
        batch=batch,
        box_vectors=box_vectors,
        use_periodic=use_periodic,
        cutoff_lower=cutoff_lower,
        cutoff_upper=cutoff_upper,
        max_num_pairs=max_num_pairs,
        loop=loop,
        include_transpose=include_transpose,
        num_cells=num_cells,
    )
