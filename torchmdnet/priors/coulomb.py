# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import OptimizedDistance, scatter
from typing import Optional, Dict


class Coulomb(BasePrior):
    """This class implements a Coulomb potential, scaled by a cosine switching function to reduce its
    effect at short distances.

    Parameters
    ----------
    lower_switch_distance : float
        distance below which the interaction strength is zero.
    upper_switch_distance : float
        distance above which the interaction has full strength
    max_num_neighbors : int
        Maximum number of neighbors per atom allowed.
    distance_scale : float, optional
        Factor to multiply with coordinates in the dataset to convert them to meters.
    energy_scale : float, optional
        Factor to multiply with energies in the dataset to convert them to Joules (*not* J/mol).
    box_vecs : torch.Tensor, optional
        Initial box vectors for periodic boundary conditions. If None, no periodic boundary conditions are used.
    dataset : Dataset
        Dataset object.

    Notes
    -----
    The Dataset used with this class must include a `partial_charges` field for each sample, and provide
    `distance_scale` and `energy_scale` attributes if they are not explicitly passed as arguments.
    """

    def __init__(
        self,
        lower_switch_distance,
        upper_switch_distance,
        max_num_neighbors,
        distance_scale=None,
        energy_scale=None,
        box_vecs=None,
        dataset=None,
    ):
        super(Coulomb, self).__init__()
        if distance_scale is None:
            distance_scale = dataset.distance_scale
        if energy_scale is None:
            energy_scale = dataset.energy_scale
        self.distance = OptimizedDistance(
            0, torch.inf, max_num_pairs=-max_num_neighbors
        )
        self.lower_switch_distance = lower_switch_distance
        self.upper_switch_distance = upper_switch_distance
        self.max_num_neighbors = max_num_neighbors
        self.distance_scale = float(distance_scale)
        self.energy_scale = float(energy_scale)
        self.initial_box = box_vecs

    def get_init_args(self):
        return {
            "lower_switch_distance": self.lower_switch_distance,
            "upper_switch_distance": self.upper_switch_distance,
            "max_num_neighbors": self.max_num_neighbors,
            "distance_scale": self.distance_scale,
            "energy_scale": self.energy_scale,
            "initial_box": self.initial_box,
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
        """Compute the Coulomb energy for each sample in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Tensor of shape (batch_size, 1) containing the energies of each sample in the batch.
        z : torch.Tensor
            Tensor of shape (num_atoms,) containing the atom types for each atom in the batch.
        pos : torch.Tensor
            Tensor of shape (num_atoms, 3) containing the positions of each atom in the batch.
        batch : torch.Tensor
            Tensor of shape (num_atoms,) containing the batch index for each atom in the batch.
        box : torch.Tensor, optional
            Tensor of shape (3, 3) containing the box vectors for the batch. If None, use the initial box vectors.
        extra_args : dict, optional
            Dictionary of extra arguments. Must contain a `partial_charges` field.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 1) containing the energies of each sample in the batch.
        """
        # Convert to nm and calculate distance.
        x = 1e9 * self.distance_scale * pos
        box = box if box is not None else self.initial_box
        edge_index, distance, _ = self.distance(x, batch, box=box)

        # Compute the energy, converting to the dataset's units.  Multiply by 0.5 because every atom pair
        # appears twice.
        q = extra_args["partial_charges"][edge_index]
        lower = torch.tensor(self.lower_switch_distance)
        upper = torch.tensor(self.upper_switch_distance)
        phase = (torch.max(lower, torch.min(upper, distance)) - lower) / (upper - lower)
        energy = (0.5 - 0.5 * torch.cos(torch.pi * phase)) * q[0] * q[1] / distance
        energy = (
            0.5
            * (2.30707e-28 / self.energy_scale / self.distance_scale)
            * scatter(energy, batch[edge_index[0]], dim=0, reduce="sum")
        )
        energy = energy.reshape(y.shape)
        return y + energy
