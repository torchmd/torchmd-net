import os
import torch as pt
from torch.utils import cpp_extension


def compile_extension():
    src_dir = os.path.dirname(__file__)
    sources = ["neighbors.cpp", "neighbors_cpu.cpp"] + (
        ["neighbors_cuda.cu", "backwards.cu"]
        if pt.cuda.is_available()
        else []
    )
    sources = [os.path.join(src_dir, name) for name in sources]
    cpp_extension.load(name="torchmdnet_neighbors", sources=sources, is_python_module=False)


def get_backends():
    compile_extension()
    get_neighbor_pairs_brute = pt.ops.torchmdnet_neighbors.get_neighbor_pairs_brute
    get_neighbor_pairs_shared = pt.ops.torchmdnet_neighbors.get_neighbor_pairs_shared
    get_neighbor_pairs_cell = pt.ops.torchmdnet_neighbors.get_neighbor_pairs_cell
    return {
        "brute": get_neighbor_pairs_brute,
        "cell": get_neighbor_pairs_cell,
        "shared": get_neighbor_pairs_shared,
    }
