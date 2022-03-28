import os
import torch.utils.cpp_extension

src_dir = os.path.dirname(__file__)
sources = ['neighbors.cpp', 'neighbors_cpu.cpp', 'neighbors_cuda.cu']
sources = [os.path.join(src_dir, name) for name in sources]

torch.utils.cpp_extension.load(name='neighbors', sources=sources, is_python_module=False)
get_neighbor_list = torch.ops.neighbors.get_neighbor_list