import torch
from torch_scatter import scatter
import numpy as np

def pow(input, power):
    if power == 1:
        return input
    elif power == 2:
        return input * input
    elif power == 3:
        return input * input * input
    elif power == 4:
        out = input * input
        return out * out
    else:
        return torch.pow(input, power)

class SparseCosineKernel(torch.nn.Module):
    def __init__(self, sparse_points, sp_map,  zeta):
        super(SparseCosineKernel, self).__init__()
        self.register_buffer("zeta", torch.tensor(zeta))
        self.register_buffer("sparse_points", sparse_points)
        self.sp_map = sp_map
        self.n_sparse_point = sparse_points.shape[0]

    def compute_KMM(self):
        KMM = []
        species = sorted(self.sp_map.keys())
        for sp in species:
            block = self.sparse_points[self.sp_map[sp]] @ self.sparse_points[self.sp_map[sp]].t()
            KMM.append(pow(block, self.zeta))
        return torch.block_diag(*KMM)

    def forward(self, PS, data):
        species = torch.unique(data['z'])
        k_mat = torch.zeros((len(data['n_atoms']), self.n_sparse_point), dtype=PS.dtype, device=PS.device)
        for sp in species.tolist():
            mask = data['z'] == sp
            k_partial = pow(PS[mask] @ self.sparse_points[self.sp_map[sp]].t(), self.zeta)
            scatter(k_partial, data['batch'][mask], dim=0, out=k_mat)
        return k_mat
