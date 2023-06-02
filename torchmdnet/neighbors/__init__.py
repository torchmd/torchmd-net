import os
import torch
from torch.utils import cpp_extension

def compile_extension():
    src_dir = os.path.dirname(__file__)
    sources = ["neighbors.cpp", "neighbors_cpu.cpp"] + (
        ["neighbors_cuda.cu", "backwards.cu"]
        if torch.cuda.is_available()
        else []
    )
    sources = [os.path.join(src_dir, name) for name in sources]
    cpp_extension.load(name="torchmdnet_neighbors", sources=sources, is_python_module=False)

def get_backends():
    compile_extension()
    get_neighbor_pairs_brute = torch.ops.torchmdnet_neighbors.get_neighbor_pairs_brute
    get_neighbor_pairs_shared = torch.ops.torchmdnet_neighbors.get_neighbor_pairs_shared
    get_neighbor_pairs_cell = torch.ops.torchmdnet_neighbors.get_neighbor_pairs_cell
    return {
        "brute": get_neighbor_pairs_brute,
        "cell": get_neighbor_pairs_cell,
        "shared": get_neighbor_pairs_shared,
    }

class NeighborKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kernel, pos, batch, max_num_pairs, cutoff_lower, cutoff_upper, loop, include_transpose, box, use_periodic, check_errors):
        edge_index, edge_vec, edge_weight, num_pairs = kernel(
            pos,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            loop=loop,
            batch=batch,
            max_num_pairs=max_num_pairs,
            include_transpose=include_transpose,
            box_vectors=box,
            use_periodic=use_periodic,
        )
        if check_errors:
            if num_pairs[0] > max_num_pairs:
                raise RuntimeError(
                    "Found num_pairs({}) > max_num_pairs({})".format(
                        num_pairs[0], max_num_pairs
                    )
                )
        edge_index = edge_index.to(torch.long)
        ctx.save_for_backward(edge_index, edge_vec, edge_weight)
        ctx.num_atoms = pos.shape[0]
        return edge_index, edge_vec, edge_weight, num_pairs

    @staticmethod
    def backward(ctx, _grad_edge_index, grad_edge_vec, grad_edge_weight, _grad_num_pairs):
        edge_index, edge_vec, edge_weight = ctx.saved_tensors
        r0 = edge_weight != 0
        self_edge = edge_index[0] == edge_index[1]
        grad_positions = torch.zeros((ctx.num_atoms,3), device=edge_vec.device, dtype=edge_vec.dtype)
        grad_edge_vec_ = grad_edge_vec.clone()
        grad_edge_vec_[self_edge] = 0
        grad_distances_ = torch.ones(edge_vec.shape, device=edge_vec.device, dtype=edge_vec.dtype)
        grad_distances_[r0] = edge_vec[r0] / edge_weight[r0].unsqueeze(-1) * grad_edge_weight[r0].unsqueeze(-1)
        grad_distances_[self_edge] = 0
        grad_positions = grad_positions.index_add(0, edge_index[0][r0], grad_edge_vec_[r0] + grad_distances_[r0])
        grad_positions = grad_positions.index_add(0, edge_index[1][r0], -grad_edge_vec_[r0]-grad_distances_[r0])
        return None, grad_positions, None, None, None, None, None, None, None, None, None
