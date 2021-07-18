import torch
import torch.nn as nn

from ..spherical_expansion import SphericalExpension


class RadialSpectrum(nn.Module):
    def __init__(self, max_radial, interaction_cutoff,
                                gaussian_sigma_constant, species, normalize=True):
        super(RadialSpectrum, self).__init__()
        self.nmax = max_radial
        self.lmax = 0
        self.rc = interaction_cutoff
        self.sigma = gaussian_sigma_constant
        self.normalize = normalize

        self.species, _ = torch.sort(species)
        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.int32)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.se = SphericalExpension(max_radial, 0, interaction_cutoff, gaussian_sigma_constant, species)

        self.D = self.n_species*self.nmax

    def forward(self, data):
        ci_an = self.se(data)

        if self.normalize:
            return torch.nn.functional.normalize(
                ci_an.view(-1, self.D), dim=1)
        else:
            return ci_an.view(-1, self.D)