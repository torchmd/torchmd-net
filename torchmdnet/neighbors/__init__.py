import os
import torch as pt
from torch.utils import cpp_extension

src_dir = os.path.dirname(__file__)
sources = ['neighbors.cpp', 'neighbors_cpu.cpp'] + (['neighbors_cuda.cu'] if pt.cuda.is_available() else [])
sources = [os.path.join(src_dir, name) for name in sources]

cpp_extension.load(name='neighbors', sources=sources, is_python_module=False)
get_neighbor_pairs = pt.ops.neighbors.get_neighbor_pairs
