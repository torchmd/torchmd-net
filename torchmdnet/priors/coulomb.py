import torch
from torchmdnet.priors.base import BasePrior
from torchmdnet.models.utils import OptimizedDistance, scatter
from typing import Optional, Dict

class Coulomb(BasePrior):
    """This class implements a Coulomb potential, scaled by erf(alpha*r) to reduce its
    effect at short distances.

    To use this prior, the Dataset must include a field called `partial_charges` with each sample,
    containing the partial charge for each atom.  It also must provide the following attributes.

    distance_scale: multiply by this factor to convert coordinates stored in the dataset to meters
    energy_scale: multiply by this factor to convert energies stored in the dataset to Joules (*not* J/mol)
    initial_box: the initial box size, in nanometers
    """
    def __init__(self, alpha, max_num_neighbors, distance_scale=None, energy_scale=None, box_vecs=None, dataset=None):
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
        self.initial_box = box_vecs
    def get_init_args(self):
        return {'alpha': self.alpha,
                'max_num_neighbors': self.max_num_neighbors,
                'distance_scale': self.distance_scale,
                'energy_scale': self.energy_scale,
                'initial_box': self.initial_box}

    def reset_parameters(self):
        pass

    def post_reduce(self, y, z, pos, batch, box: Optional[torch.Tensor] = None, extra_args: Optional[Dict[str, torch.Tensor]] = None):
        # Convert to nm and calculate distance.
        x = 1e9*self.distance_scale*pos
        alpha = self.alpha/(1e9*self.distance_scale)
        box = box if box is not None else self.initial_box
        edge_index, distance, _ = self.distance(x, batch, box=box)

        # Compute the energy, converting to the dataset's units.  Multiply by 0.5 because every atom pair
        # appears twice.
        q = extra_args['partial_charges'][edge_index]
        energy = torch.erf(alpha*distance)*q[0]*q[1]/distance
        energy = 0.5*(2.30707e-28/self.energy_scale/self.distance_scale)*scatter(energy, batch[edge_index[0]], dim=0, reduce="sum")
        energy = energy.reshape(y.shape)
        return y + energy
