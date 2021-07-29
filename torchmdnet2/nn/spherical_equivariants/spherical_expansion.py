import torch
import torch.nn as nn

from e3nn import o3
from torch_scatter import scatter

from ..radial_basis import splined_radial_integrals
from ..cutoff import ShiftedCosineCutoff
from ...neighbor_list import torch_neighbor_list

def mult_ln_lm(x_ln, x_lm):
    idx = 0
    J, lmax, nmax = x_ln.shape
    _, lmmax = x_lm.shape
    x_out = torch.zeros((J, nmax, lmmax), dtype=x_lm.dtype)
    for l in range(lmax):
        l_block_length = 2*l+1
        x_out[...,idx:(idx+l_block_length)] = torch.einsum('jn,jm->jnm',
                                                           x_ln[:,l,:], x_lm[:, idx:(idx+l_block_length)])

        idx += l_block_length

    return x_out

def species_dependant_reduction_ids(data, species2idx):
    zj = data.z[data.idx_j]
    ids = torch.zeros_like(data.idx_i)
    n_species = len(torch.unique(species2idx))-1
    for ii, sp_j in enumerate(zj):
        ids[ii] = species2idx[sp_j]+ n_species*data.idx_i[ii]
    return ids

class SphericalExpansion(nn.Module):
    """
    Computes the spherical expansion of an atomic density. For a given atom :math:`i` in an atomic structure, the expansion coefficients are computed as:

    .. math::

        c^i_{anlm} = \sum_{(j,b) \in i} \delta_{ab} I_{nl}(r_{ij} Y_l^m(\hat{\mathbf{r}}_{ij})) f_c(r_{ij})

    where the sum runs over the neighbors :math:`j` of atom i with atomic type :math:`b`, :math:`f_c` is a cutoff function, :math:`I_{nl}` is the radial integral and :math:`Y_l^m` are spherical harmonics.

    """
    def __init__(self, nmax, lmax, rc, sigma, species, smooth_width=0.5):
        super(SphericalExpansion, self).__init__()
        self.nmax = nmax
        self.lmax = lmax
        self.rc = rc
        self.sigma = sigma
        self.cutoff = ShiftedCosineCutoff(rc, smooth_width)

        self.species, _ = torch.sort(species)
        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.int32)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.Rln = splined_radial_integrals(self.nmax, self.lmax+1,
                                self.rc, self.sigma, mesh_size=600)

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.Ylm = o3.SphericalHarmonics(self.irreps_sh, normalization='integral',
                                            normalize=False)

    def forward(self, data):
        if 'direction_vectors' not in data or 'distances' not in data:
            idx_i, idx_j, cell_shifts = torch_neighbor_list(data, self.rc, self_interaction=True)
            rij = (data.pos[idx_j] - data.pos[idx_i] + cell_shifts)
            data.distances = rij.norm(dim=1).view(-1, 1)
            data.direction_vectors = torch.nn.functional.normalize(rij, dim=1)
            data.idx_i = idx_i
            data.idx_j = idx_j

        reduction_ids = species_dependant_reduction_ids(data, self.species2idx)
        Ylm = self.Ylm(data.direction_vectors)
        RIln = self.Rln(data.distances).view(-1,self.lmax+1,self.nmax) * self.cutoff(data.distances)[:, None, None]
        n_atoms = torch.sum(data.n_atoms)
        # return Ylm, RIln
        cij_nlm = mult_ln_lm(RIln, Ylm)
        ci_anlm = scatter(cij_nlm, reduction_ids,
                            dim_size=self.n_species*n_atoms, dim=0, reduce='sum')

        return ci_anlm.view(n_atoms, self.n_species, self.nmax, (self.lmax+1)**2)
