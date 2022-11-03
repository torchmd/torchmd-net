from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import Distance
import torch as pt
from torch_scatter import scatter


class D2(BasePrior):
    """Dispersive correction term as used in DFT-D2

    Reference
    ---------
    Grimme, Stefan. "Semiempirical GGA-type density functional constructed with a long‐range dispersion correction." Journal of computational chemistry 27.15 (2006): 1787-1799.
    https://onlinelibrary.wiley.com/doi/10.1002/jcc.20495

    Arguments
    ---------
    cutoff: float
        Distance cutoff for the correction term
    max_num_neighbors: int
        Maximum number of neighbors
    atomic_numbers: list of ints or None
        Map of atom types to atomic numbers.
        If `atomic_numbers=None`, use `dataset.atomic_numbers`
    position_scale: float or None
        Multiply by this factor to convert positions stored in the dataset to meters (m).
        If `position_scale=None` (default), use `dataset.position_scale`
    energy_scale: float or None
        Multiply by this factor to convert energies stored in the dataset to Joules (J).
        Note: *not* J/mol.
        If `energy_scale=None` (default), use `dataset.energy_scale`
    dataset: Dataset or None
        Dataset object.
        If `dataset=None`; `atomic_numbers`, `position_scale`, and `energy_scale` have to be set.
    """

    # Taken from Table 1
    # C_6 parameters (J/mol*nm^6) and and van der Waals radii (Å) for elements
    C_6_R_r = pt.tensor(
        [
            [pt.nan, pt.nan],  # 0
            [0.14, 1.001],  # 1 H
            [0.08, 1.012],  # 2 He
            [1.61, 0.825],  # 3 Li
            [1.61, 1.408],  # 4 Be
            [3.13, 1.485],  # 5 B
            [1.75, 1.452],  # 6 C
            [1.23, 1.397],  # 7 N
            [0.70, 1.342],  # 8 O
            [0.75, 1.287],  # 9 F
            [0.63, 1.243],  # 10 Ne
            # Values up to Xe are available
        ]
    )
    C_6_R_r[:, 1] *= 0.1  # Å --> nm

    def __init__(
        self,
        cutoff,
        max_num_neighbors,
        atomic_numbers=None,
        position_scale=None,
        energy_scale=None,
        dataset=None,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)

        self.atomic_numbers = list(
            dataset.atomic_numbers if atomic_numbers is None else atomic_numbers
        )
        self.position_scale = float(
            dataset.position_scale if position_scale is None else position_scale
        )
        self.energy_scale = float(
            dataset.energy_scale if energy_scale is None else energy_scale
        )

        self.position_scale *= 1e9  # m --> nm
        self.energy_scale *= 6.02214076e23  # J --> J/mol

        self.distances = Distance(
            cutoff_lower=0,
            cutoff_upper=self.cutoff / self.position_scale,
            max_num_neighbors=self.max_num_neighbors,
        )

        # Atomic number mapping
        self.register_buffer("Z_map", pt.tensor(self.atomic_numbers, dtype=pt.long))

        # Atomic parameters
        self.register_buffer("C_6", self.C_6_R_r[:, 0])
        self.register_buffer("R_r", self.C_6_R_r[:, 1])

        # Default values from the reference
        self.d = 20
        self.s_6 = 1

    def get_init_args(self):
        return {
            "cutoff": self.cutoff,
            "max_num_neighbors": self.max_num_neighbors,
            "atomic_numbers": self.atomic_numbers,
            "position_scale": self.position_scale,
            "energy_scale": self.energy_scale,
        }

    def post_reduce(self, y, z, pos, batch):

        ij, R_ij, _ = self.distances(pos * self.position_scale, batch)

        Z = self.Z_map[z[ij]]
        C_6 = self.C_6[Z].prod(dim=0).sqrt()
        R_r = self.R_r[Z].sum(dim=0)

        f_dump = 1 / (1 + pt.exp(-self.d * (R_ij / R_r - 1)))
        E_ij = C_6 / R_ij**6 * f_dump

        batch = batch[ij[0]]
        E_disp = -self.s_6 * scatter(E_ij, batch, dim=0, reduce="sum")
        E_disp /= 2

        return y + E_disp / self.energy_scale
