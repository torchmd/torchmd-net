# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import OptimizedDistance, CosineCutoff, scatter
from typing import Optional, Dict


class ZBL(BasePrior):
    """
    Implements the Ziegler-Biersack-Littmark (ZBL) potential for screened nuclear repulsion.

    This potential is described in Ziegler, J.F., Biersack, J.P., Littmark, U. "The Stopping and Range of Ions in Solids."
    (1985), specifically in equations 9 and 10 on page 147. It is an empirical potential effectively describing the
    repulsion between atoms at very short distances.

    Reference:
        Available at: https://doi.org/10.1007/978-3-642-68779-2_5

    Parameters
    ----------
    atomic_number : torch.Tensor, optional
        A 1D tensor of length max_z. `atomic_number[z]` is the atomic number of atoms with atom type z. If None, use `dataset.atomic_number`.
    distance_scale : float, optional
        Factor to multiply with coordinates stored in the dataset to convert them to meters. If None, use `dataset.distance_scale`.
    energy_scale : float, optional
        Factor to multiply with energies stored in the dataset to convert them to Joules (not J/mol). If None, use `dataset.energy_scale`.
    dataset : Dataset, optional
        Dataset object. If None, `atomic_number`, `distance_scale`, and `energy_scale` must be explicitly set.

    """

    def __init__(
        self,
        cutoff_distance,
        max_num_neighbors,
        atomic_number=None,
        distance_scale=None,
        energy_scale=None,
        dataset=None,
    ):
        super(ZBL, self).__init__()
        if atomic_number is None:
            atomic_number = dataset.atomic_number
        if distance_scale is None:
            distance_scale = dataset.distance_scale
        if energy_scale is None:
            energy_scale = dataset.energy_scale
        atomic_number = torch.as_tensor(atomic_number, dtype=torch.long)
        self.register_buffer("atomic_number", atomic_number)
        self.distance = OptimizedDistance(
            0, cutoff_distance, max_num_pairs=-max_num_neighbors
        )
        self.cutoff = CosineCutoff(cutoff_upper=cutoff_distance)
        self.cutoff_distance = cutoff_distance
        self.max_num_neighbors = max_num_neighbors
        self.distance_scale = float(distance_scale)
        self.energy_scale = float(energy_scale)

    def get_init_args(self):
        return {
            "cutoff_distance": self.cutoff_distance,
            "max_num_neighbors": self.max_num_neighbors,
            "atomic_number": self.atomic_number.tolist(),
            "distance_scale": self.distance_scale,
            "energy_scale": self.energy_scale,
        }

    def reset_parameters(self):
        pass

    def post_reduce(
        self,
        y,
        z,
        pos,
        batch,
        box: Optional[torch.Tensor] = None,
        extra_args: Optional[Dict[str, torch.Tensor]] = None,
    ):
        edge_index, distance, _ = self.distance(pos, batch, box)
        if edge_index.shape[1] == 0:
            return y
        atomic_number = self.atomic_number[z[edge_index]]
        # 5.29e-11 is the Bohr radius in meters.  All other numbers are magic constants from the ZBL potential.
        a = (
            0.8854
            * 5.29177210903e-11
            / (atomic_number[0] ** 0.23 + atomic_number[1] ** 0.23)
        )
        d = distance * self.distance_scale / a
        f = (
            0.1818 * torch.exp(-3.2 * d)
            + 0.5099 * torch.exp(-0.9423 * d)
            + 0.2802 * torch.exp(-0.4029 * d)
            + 0.02817 * torch.exp(-0.2016 * d)
        )
        f *= self.cutoff(distance)
        # Compute the energy, converting to the dataset's units.  Multiply by 0.5 because every atom pair
        # appears twice.
        energy = f * atomic_number[0] * atomic_number[1] / distance
        energy = (
            0.5
            * (2.30707755e-28 / self.energy_scale / self.distance_scale)
            * scatter(energy, batch[edge_index[0]], dim=0, reduce="sum")
        )
        if energy.shape[0] < y.shape[0]:
            energy = torch.nn.functional.pad(energy, (0, y.shape[0] - energy.shape[0]))
        energy = energy.reshape(y.shape)
        return y + energy
