import torch.utils.cpp_extension

torch.utils.cpp_extension.load(name='neighbors', sources=['neighbors.cpp', 'neighbors_cpu.cpp', 'neighbors_cuda.cu'], is_python_module=False)
get_neighbor_list = torch.ops.neighbors.get_neighbor_list