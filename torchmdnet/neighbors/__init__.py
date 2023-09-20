import torch
torch.ops.load_library("neighbors.so")
get_neighbor_pairs_kernel = torch.ops.neighbors.get_neighbor_pairs
