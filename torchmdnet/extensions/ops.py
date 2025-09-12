# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

# Place here any short extensions to torch that you want to use in your code.
# The extensions present in extensions.cpp will be automatically compiled in setup.py and loaded here.
# The extensions will be available under torch.ops.torchmdnet_extensions, but you can add wrappers here to make them more convenient to use.
# Place here too any meta registrations for your extensions if required.

import torch
from torch import Tensor
from typing import Tuple


__all__ = ["is_current_stream_capturing", "get_neighbor_pairs_kernel"]


from torch.utils.cpp_extension import load


def set_torch_cuda_arch_list():
    """Set the CUDA arch list according to the architectures the current torch installation was compiled for.
    This function is a no-op if the environment variable TORCH_CUDA_ARCH_LIST is already set or if torch was not compiled with CUDA support.
    """
    import os
    import sys

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        arch_flags = torch._C._cuda_getArchFlags()
        sm_versions = [x[3:] for x in arch_flags.split() if x.startswith("sm_")]
        formatted_versions = ";".join([f"{y[:-1]}.{y[-1]}" for y in sm_versions])
        formatted_versions += "+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = formatted_versions
    # Find where the python interpreter is installed
    python_interpreter = sys.executable
    CUDA_HOME = os.path.dirname(os.path.dirname(python_interpreter))
    print(f"CUDA_HOME: {CUDA_HOME}")
    os.environ["CUDA_HOME"] = CUDA_HOME


sources = [
    "torchmdnet/extensions/torchmdnet_extensions.cpp",
    "torchmdnet/extensions/neighbors/neighbors_cpu.cpp",
]
if torch.cuda.is_available():
    set_torch_cuda_arch_list()
    sources.append("torchmdnet/extensions/neighbors/neighbors_cuda.cu")


torchmdnet_extensions = load(
    name="torchmdnet_extensions",
    sources=sources,
    extra_cuda_cflags=["-O3", "-lineinfo"],  # optional
    verbose=True,
)


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
    """Computes the neighbor pairs for a given set of atomic positions.
    The list is generated as a list of pairs (i,j) without any enforced ordering.
    The list is padded with -1 to the maximum number of pairs.

    Parameters
    ----------
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.
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


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torchmdnet_extensions::get_neighbor_pairs_bkwd")
def _(
    grad_edge_vec: Tensor,
    grad_edge_weight: Tensor,
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_weight: Tensor,
    num_atoms: int,
):
    return torch.zeros((num_atoms, 3), dtype=edge_vec.dtype, device=edge_vec.device)


@torch.library.register_fake("torchmdnet_extensions::get_neighbor_pairs_fwd")
def _(
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
    """Returns empty vectors with the correct shape for the output of get_neighbor_pairs_kernel."""
    size = max_num_pairs
    edge_index = torch.empty((2, size), dtype=torch.long, device=positions.device)
    edge_distance = torch.empty((size,), dtype=positions.dtype, device=positions.device)
    edge_vec = torch.empty((size, 3), dtype=positions.dtype, device=positions.device)
    num_pairs = torch.empty((1,), dtype=torch.long, device=positions.device)
    return edge_index, edge_vec, edge_distance, num_pairs
