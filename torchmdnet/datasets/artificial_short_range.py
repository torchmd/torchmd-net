import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


def random_vectors_in_sphere_box_muller(radius, count):
    # Generate uniformly distributed random numbers for Box-Muller
    u1 = np.random.uniform(low=0.0, high=1.0, size=count)
    u2 = np.random.uniform(low=0.0, high=1.0, size=count)
    u3 = np.random.uniform(low=0.0, high=1.0, size=count)

    # Box-Muller transform for normal distribution
    normal1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    normal2 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
    normal3 = np.sqrt(-2.0 * np.log(u3)) * np.cos(
        2.0 * np.pi * u2
    )  # Using u2 again for the third component

    # Stack the normals
    vectors = np.column_stack((normal1, normal2, normal3))

    # Normalize each vector to have magnitude 1
    norms = np.linalg.norm(vectors, axis=1)
    vectors_normalized = vectors / norms[:, np.newaxis]

    # Scale vectors by random radii up to 'radius'
    scale = np.random.uniform(0, radius**3, count) ** (
        1 / 3
    )  # Cube root to ensure uniform distribution in volume
    vectors_scaled = vectors_normalized * scale[:, np.newaxis]

    return vectors_scaled


def compute_energy(pos, max_dist):
    dist = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)  # shape (size,)
    y = 20 + 80 * (1 - dist / max_dist)  # shape (size,)
    return y


def compute_forces(pos, max_dist):
    pos = pos.clone().detach().requires_grad_(True)
    y = compute_energy(pos, max_dist)
    y_sum = y.sum()
    y_sum.backward()
    forces = -pos.grad
    return forces


class ShortRange(Dataset):
    def __init__(self, root, max_dist, size, max_z, transform=None, pre_transform=None):
        super(ShortRange, self).__init__(root, transform, pre_transform)
        self.max_dist = max_dist
        self.size = size
        self.max_z = max_z
        # Create some npy files with random data. The dataset consists of pairs of atoms, with their positions, atomic numbers and energy
        # Positions inside a sphere of radius max_dist
        self.pos = random_vectors_in_sphere_box_muller(max_dist, 2 * size)
        self.pos = self.pos.reshape(size, 2, 3)
        # Atomic numbers
        self.z = np.random.randint(1, max_z, size=2 * size).reshape(size, 2)
        # Energy
        self.y = compute_energy(torch.tensor(self.pos), max_dist).detach().numpy() * 0
        assert self.y.shape == (size,)
        assert self.z.shape == (size, 2)
        # Negative gradient of the energy with respect to the positions, should have the same shape as pos
        self.neg_dy = (
            compute_forces(torch.tensor(self.pos, dtype=torch.float), max_dist)
            .detach()
            .numpy()
            * 0
        )

    def get(self, idx):
        y = torch.tensor(self.y[idx], dtype=torch.float).view(1, 1)
        z = torch.tensor(self.z[idx], dtype=torch.long).view(2)
        pos = torch.tensor(self.pos[idx], dtype=torch.float).view(2, 3)
        neg_dy = torch.tensor(self.neg_dy[idx], dtype=torch.float).view(2, 3)
        data = Data(
            z=z,
            pos=pos,
            y=y,
            neg_dy=neg_dy,
        )
        return data

    def len(self):
        return self.size

        # Taken from https://github.com/isayev/ASE_ANI/blob/master/ani_models/ani-2x_8x/sae_linfit.dat

    _ELEMENT_ENERGIES = {
        1: -0.5978583943827134,  # H
        6: -38.08933878049795,  # C
        7: -54.711968298621066,  # N
        8: -75.19106774742086,  # O
        9: -99.80348506781634,  # F
        16: -398.1577125334925,  # S
        17: -460.1681939421027,  # Cl
    }
    HARTREE_TO_EV = 27.211386246  #::meta private:

    def get_atomref(self, max_z=100):
        """Atomic energy reference values for the :py:mod:`torchmdnet.priors.Atomref` prior.

        Args:
            max_z (int): Maximum atomic number

        Returns:
            torch.Tensor: Atomic energy reference values for each element in the dataset.
        """
        refs = torch.zeros(max_z)
        for key, val in self._ELEMENT_ENERGIES.items():
            refs[key] = val * self.HARTREE_TO_EV * 0

        return refs.view(-1, 1)
