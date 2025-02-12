# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import OptimizedDistance, scatter
import torch as pt
from typing import Optional, Dict


class D2(BasePrior):
    """
    Dispersive correction term as used in DFT-D2.

    Reference:
        Grimme, Stefan. "Semiempirical GGA-type density functional constructed with a long‐range dispersion correction."
        Journal of computational chemistry 27.15 (2006): 1787-1799.
        Available at: https://onlinelibrary.wiley.com/doi/10.1002/jcc.20495

    Parameters
    ----------
    cutoff_distance : float
        Distance cutoff for the correction term.
    max_num_neighbors : int
        Maximum number of neighbors to consider.
    atomic_number : list of int, optional
        Map of atom types to atomic numbers. If None, use `dataset.atomic_numbers`.
    position_scale : float, optional
        Factor to convert positions stored in the dataset to meters (m). If None (default), use `dataset.position_scale`.
    energy_scale : float, optional
        Factor to convert energies stored in the dataset to Joules (J). Note: not J/mol. If None (default), use `dataset.energy_scale`.
    dataset : Dataset, optional
        Dataset object. If None, `atomic_number`, `position_scale`, and `energy_scale` must be explicitly set.

    Examples
    --------
    >>> from torchmdnet.priors import D2
    >>> prior = D2(
            cutoff_distance=10.0,  # Å
            max_num_neighbors=128,
            atomic_number=list(range(100)),
            position_scale=1e-10,  # Å --> m
            energy_scale=4.35974e-18,  # Hartree --> J
        )
    """

    # C_6 parameters (J/mol*nm^6) and and van der Waals radii (Å) for elements.
    # Taken from Table 1
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
            [5.71, 1.144],  # 11 Na
            [5.71, 1.364],  # 12 Mg
            [10.79, 1.639],  # 13 Al
            [9.23, 1.716],  # 14 Si
            [7.84, 1.705],  # 15 P
            [5.57, 1.683],  # 16 S
            [5.07, 1.639],  # 17 Cl
            [4.61, 1.595],  # 18 Ar
            [10.80, 1.485],  # 19 K
            [10.80, 1.474],  # 20 Ca
            [10.80, 1.562],  # 21 Sc
            [10.80, 1.562],  # 22 Ti
            [10.80, 1.562],  # 23 V
            [10.80, 1.562],  # 24 Cr
            [10.80, 1.562],  # 25 Mn
            [10.80, 1.562],  # 26 Fe
            [10.80, 1.562],  # 27 Co
            [10.80, 1.562],  # 28 Ni
            [10.80, 1.562],  # 29 Cu
            [10.80, 1.562],  # 30 Zn
            [16.99, 1.650],  # 31 Ga
            [17.10, 1.727],  # 32 Ge
            [16.37, 1.760],  # 33 As
            [12.64, 1.771],  # 34 Se
            [12.47, 1.749],  # 35 Br
            [12.01, 1.727],  # 36 Kr
            [24.67, 1.628],  # 37 Rb
            [24.67, 1.606],  # 38 Sr
            [24.67, 1.639],  # 39 Y
            [24.67, 1.639],  # 40 Zr
            [24.67, 1.639],  # 41 Nb
            [24.67, 1.639],  # 42 Mo
            [24.67, 1.639],  # 43 Tc
            [24.67, 1.639],  # 44 Ru
            [24.67, 1.639],  # 45 Rh
            [24.67, 1.639],  # 46 Pd
            [24.67, 1.639],  # 47 Ag
            [24.67, 1.639],  # 48 Cd
            [37.32, 1.672],  # 49 In
            [38.71, 1.804],  # 50 Sn
            [38.44, 1.881],  # 51 Sb
            [31.74, 1.892],  # 52 Te
            [31.50, 1.892],  # 53 I
            [29.99, 1.881],  # 54 Xe
        ],
        dtype=pt.float64,
    )  #::meta private:
    C_6_R_r[:, 1] *= 0.1  # Å --> nm

    def __init__(
        self,
        cutoff_distance,
        max_num_neighbors,
        atomic_number=None,
        distance_scale=None,
        energy_scale=None,
        dataset=None,
        dtype=pt.float32,
    ):
        super().__init__()
        one = pt.tensor(1.0, dtype=dtype).item()
        self.cutoff_distance = cutoff_distance * one
        self.max_num_neighbors = int(max_num_neighbors)

        self.C_6_R_r = self.C_6_R_r.to(dtype=dtype)
        self.atomic_number = list(
            dataset.atomic_number if atomic_number is None else atomic_number
        )
        self.distance_scale = one * (
            dataset.distance_scale if distance_scale is None else distance_scale
        )
        self.energy_scale = one * (
            dataset.energy_scale if energy_scale is None else energy_scale
        )

        # Distance calculator
        self.distances = OptimizedDistance(
            cutoff_lower=0,
            cutoff_upper=self.cutoff_distance,
            max_num_pairs=-self.max_num_neighbors,
        )

        # Parameters (default values from the reference)
        self.register_buffer("Z_map", pt.tensor(self.atomic_number, dtype=pt.long))
        self.register_buffer("C_6", self.C_6_R_r[:, 0])
        self.register_buffer("R_r", self.C_6_R_r[:, 1])
        self.d = 20
        self.s_6 = 1

    def reset_parameters(self):
        pass

    def get_init_args(self):
        return {
            "cutoff_distance": self.cutoff_distance,
            "max_num_neighbors": self.max_num_neighbors,
            "atomic_number": self.atomic_number,
            "distance_scale": self.distance_scale,
            "energy_scale": self.energy_scale,
        }

    def post_reduce(
        self,
        y,
        z,
        pos,
        batch,
        box: Optional[pt.Tensor] = None,
        extra_args: Optional[Dict[str, pt.Tensor]] = None,
    ):

        # Convert to interal units: nm and J/mol
        # NOTE: float32 is overflowed, if m and J are used
        distance_scale = self.distance_scale * 1e9  # m --> nm
        energy_scale = self.energy_scale * 6.02214076e23  # J --> J/mol

        # Get atom pairs and their distancence
        ij, R_ij, _ = self.distances(pos, batch)
        R_ij *= distance_scale

        # No interactions
        if ij.shape[1] == 0:
            return y

        # Compute the pair parameters
        Z = self.Z_map[z[ij]]
        C_6 = self.C_6[Z].prod(dim=0).sqrt()
        R_r = self.R_r[Z].sum(dim=0)

        # Compute pair contributions
        f_damp = 1 / (1 + pt.exp(-self.d * (R_ij / R_r - 1)))
        E_ij = C_6 / R_ij**6 * f_damp

        # Acculate the contributions
        batch = batch[ij[0]]
        E_disp = -self.s_6 * scatter(E_ij, batch, dim=0, reduce="sum")
        E_disp /= 2  # The pairs appear twice
        E_disp = E_disp.reshape(y.shape)

        return y + E_disp / energy_scale
