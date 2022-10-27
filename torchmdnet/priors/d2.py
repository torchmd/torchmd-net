from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import Distance
import torch as pt
from torch_scatter import scatter

# Taken from Table 1. Units: J/mol*nm^6
C_6 = pt.tensor(
    [
        pt.nan,  # 0
        0.14,  # 1 H
        0.08,  # 2 He
        1.61,  # 3 Li
        1.61,  # 4 Be
        3.13,  # 5 B
        1.75,  # 6 C
        1.23,  # 7 N
        0.70,  # 8 O
        0.75,  # 9 F
        0.63,  # 10 Ne
        # Values up to Xe are available
    ]
)
C_6 *= 1.03643e-5 * 10**6  # J/mol*nm^6 --> eV*Å^6

# Taken from Table 1. Units: Å
R_r = pt.tensor(
    [
        pt.nan,  # 0
        1.001,  # 1 H
        1.012,  # 2 He
        0.825,  # 3 Li
        1.408,  # 4 Be
        1.485,  # 5 B
        1.452,  # 6 C
        1.397,  # 7 N
        1.342,  # 8 O
        1.287,  # 9 F
        1.243,  # 10 Ne
        # Values up to Xe are available
    ]
)


class D2(BasePrior):
    """Dispersive correction term as used in DFT-D2

    Units:
      - Atomic postions: Å
      - Energies: eV

    Reference: Grimme, Stefan. "Semiempirical GGA-type density functional
               constructed with a long‐range dispersion correction."
               Journal of computational chemistry 27.15 (2006): 1787-1799.
               https://onlinelibrary.wiley.com/doi/10.1002/jcc.20495
    """

    def __init__(self):
        super().__init__()

        self.distances = Distance(
            cutoff_lower=0, cutoff_upper=12.0, max_num_neighbors=128
        )

        # Atomic parameters
        self.register_buffer("C_6", C_6)
        self.register_buffer("R_r", R_r)

        # Default values from the reference
        self.d = 20
        self.s_6 = 1

    def post_reduce(self, y, z, pos, batch):

        ij, R_ij, _ = self.distances(pos, batch)

        Z = z[ij]
        C_6 = self.C_6[Z].prod(dim=0).sqrt()
        R_r = self.R_r[Z].sum(dim=0)

        f_dump = 1 / (1 + pt.exp(-self.d * (R_ij / R_r - 1)))
        E_ij = C_6 / R_ij**6 * f_dump

        batch = batch[ij[0]]
        E_disp = -self.s_6 * scatter(E_ij, batch, dim=0, reduce="sum")
        E_disp /= 2

        return y + E_disp
