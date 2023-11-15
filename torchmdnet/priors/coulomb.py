import torch
from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import OptimizedDistance, scatter
from typing import Optional, Dict

class Coulomb(BasePrior):
    """This class implements a Coulomb potential, scaled by :math:`\\textrm{erf}(\\textrm{alpha}*r)` to reduce its
    effect at short distances.

    Parameters
    ----------
    alpha : float
        Scaling factor for the error function.
    max_num_neighbors : int
        Maximum number of neighbors per atom allowed.
    distance_scale : float, optional
        Factor to multiply with coordinates in the dataset to convert them to meters.
    energy_scale : float, optional
        Factor to multiply with energies in the dataset to convert them to Joules (*not* J/mol).
    dataset : Dataset
        Dataset object.

    Notes
    -----
    The Dataset used with this class must include a `partial_charges` field for each sample, and provide
    `distance_scale` and `energy_scale` attributes if they are not explicitly passed as arguments.
    """
    def __init__(self, alpha, max_num_neighbors, distance_scale=None, energy_scale=None, dataset=None):
        super(Coulomb, self).__init__()
        if distance_scale is None:
            distance_scale = dataset.distance_scale
        if energy_scale is None:
            energy_scale = dataset.energy_scale
        self.distance = OptimizedDistance(0, torch.inf, max_num_pairs=-max_num_neighbors)
        self.alpha = alpha
        self.max_num_neighbors = max_num_neighbors
        self.distance_scale = float(distance_scale)
        self.energy_scale = float(energy_scale)

    def get_init_args(self):
        return {'alpha': self.alpha,
                'max_num_neighbors': self.max_num_neighbors,
                'distance_scale': self.distance_scale,
                'energy_scale': self.energy_scale}

    def reset_parameters(self):
        pass

    def post_reduce(self, y, z, pos, batch, extra_args: Optional[Dict[str, torch.Tensor]]):
        # Convert to nm and calculate distance.
        x = 1e9*self.distance_scale*pos
        alpha = self.alpha/(1e9*self.distance_scale)
        edge_index, distance, _ = self.distance(x, batch)

        # Compute the energy, converting to the dataset's units.  Multiply by 0.5 because every atom pair
        # appears twice.
        q = extra_args['partial_charges'][edge_index]
        energy = torch.erf(alpha*distance)*q[0]*q[1]/distance
        energy = 0.5*(2.30707e-28/self.energy_scale/self.distance_scale)*scatter(energy, batch[edge_index[0]], dim=0, reduce="sum")
        energy = energy.reshape(y.shape)
        return y + energy
