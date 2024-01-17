# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

# Place here any short extensions to torch that you want to use in your code.
# The extensions present in extensions.cpp will be automatically compiled in setup.py and loaded here.
# The extensions will be available under torch.ops.torchmdnet_extensions, but you can add wrappers here to make them more convenient to use.
import os.path as osp
import torch
import importlib.machinery
from typing import Tuple


def _load_library(library):
    """Load a dynamic library containing torch extensions from the given path.
    Args:
        library (str): The name of the library to load.
    """
    # Find the specification for the library
    spec = importlib.machinery.PathFinder().find_spec(library, [osp.dirname(__file__)])
    # Check if the specification is found and load the library
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(
            f"Could not find module '{library}' in {osp.dirname(__file__)}"
        )


_load_library("torchmdnet_extensions")


def is_current_stream_capturing():
    """Returns True if the current CUDA stream is capturing.

    Returns False if CUDA is not available or the current stream is not capturing.

    This utility is required because the builtin torch function that does this is not scriptable.
    """
    _is_current_stream_capturing = (
        torch.ops.torchmdnet_extensions.is_current_stream_capturing
    )
    return _is_current_stream_capturing()


def get_neighbor_pairs_kernel(
    strategy: str,
    positions: torch.Tensor,
    batch: torch.Tensor,
    box_vectors: torch.Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the neighbor pairs for a given set of atomic positions.

    The list is generated as a list of pairs (i,j) without any enforced ordering.
    The list is padded with -1 to the maximum number of pairs.

    Parameters
    ----------
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.
    positions : torch.Tensor
        A tensor with shape (N, 3) representing the atomic positions.
    batch : torch.Tensor
        A tensor with shape (N,). Specifies the batch for each atom.
    box_vectors : torch.Tensor
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

    Returns
    -------
    neighbors : torch.Tensor
        List of neighbors for each atom. Shape (2, max_num_pairs).
    distances : torch.Tensor
        List of distances for each atom. Shape (max_num_pairs,).
    distance_vecs : torch.Tensor
        List of distance vectors for each atom. Shape (max_num_pairs, 3).
    num_pairs : torch.Tensor
        The number of pairs found.

    Notes
    -----
    This function is a torch extension loaded from `torch.ops.torchmdnet_extensions.get_neighbor_pairs`.
    """
    return torch.ops.torchmdnet_extensions.get_neighbor_pairs(
        strategy,
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


# For some unknown reason torch.compile is not able to compile this function
if int(torch.__version__.split(".")[0]) >= 2:
    import torch._dynamo as dynamo
    dynamo.disallow_in_graph(torch.ops.torchmdnet_extensions.get_neighbor_pairs)
