import os
import torch
from torch.utils import cpp_extension

def compile_extension():
    src_dir = os.path.dirname(__file__)
    sources = ["neighbors.cpp", "neighbors_cpu.cpp"] + (
        ["neighbors_cuda.cu"] if torch.cuda.is_available() else []
    )
    sources = [os.path.join(src_dir, name) for name in sources]
    cpp_extension.load(
        name="torchmdnet_neighbors", sources=sources, is_python_module=False
    )

compile_extension()
get_neighbor_pairs_kernel = torch.ops.torchmdnet_neighbors.get_neighbor_pairs
