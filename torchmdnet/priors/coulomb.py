import torch
from torchmdnet.priors.base import BasePrior
from torch_scatter import scatter
from torchmdnet.models.utils import OptimizedDistance
from typing import Optional, Dict

class Coulomb(BasePrior):
    """This class implements a Coulomb potential, scaled by erf(alpha*r) to reduce its
    effect at short distances.

    To use this prior, the Dataset must include a field called `partial_charges` with each sample,
    containing the partial charge for each atom.  It also must provide the following attributes.

    distance_scale: multiply by this factor to convert coordinates stored in the dataset to meters
    energy_scale: multiply by this factor to convert energies stored in the dataset to Joules (*not* J/mol)
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
