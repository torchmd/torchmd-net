import torch
import torch.nn as nn
import math
from ..spherical_expansion import SphericalExpansion

def powerspectrum(se_, nsp, nmax, lmax):
    J = se_.shape[0]
    se = se_.view((J, nsp, nmax, lmax**2))
    ps = torch.zeros(J, nsp, nmax, nsp, nmax, lmax, dtype=se.dtype, device=se.device)
    idx = 0
    for l in range(lmax):
        lbs = 2*l+1
        ps[..., l] = torch.sum(torch.einsum('ianl,ibml->ianbml', se[..., idx:idx+lbs], se[..., idx:idx+lbs]),
                                   dim=5) / math.sqrt(lbs)
        idx += lbs
    ps = ps.view(J, nsp* nmax, nsp* nmax, lmax)
    PS = torch.zeros(J, int((nsp* nmax+1)**2/2), lmax, dtype=se.dtype, device=se.device)

    fac = math.sqrt(2.) * torch.ones((nsp*nmax,nsp*nmax))
    fac[range(nsp*nmax), range(nsp*nmax)] = 1.
    ids = [(i,j) for i in range(nsp*nmax)
                for j in range(nsp*nmax) if j >= i]
    for ii, (i, j) in enumerate(ids):
            PS[:, ii, :] = fac[i,j]*ps[:, i, j, :]
    return PS.view(J, -1)

class PowerSpectrum(nn.Module):
    def __init__(self, max_radial, max_angular, interaction_cutoff,
                                gaussian_sigma_constant, species, normalize=True, smooth_width=0.5):
        super(PowerSpectrum, self).__init__()
        self.nmax = max_radial
        self.lmax = max_angular
        self.rc = interaction_cutoff
        self.sigma = gaussian_sigma_constant
        self.normalize = normalize

        self.species, _ = torch.sort(species)
        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.int32)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.se = SphericalExpansion(max_radial, max_angular, interaction_cutoff, gaussian_sigma_constant, species, smooth_width=smooth_width)

        self.D = int((self.n_species*self.nmax+1)**2/2) * (self.lmax+1)

    def forward(self, data):
        ci_anlm = self.se(data)
        pi_anbml = powerspectrum(ci_anlm, self.n_species,
                                        self.nmax, self.lmax+1)
        if self.normalize:
            return torch.nn.functional.normalize(
                pi_anbml.view(-1, self.D), dim=1)
        else:
            return pi_anbml.view(-1, self.D)