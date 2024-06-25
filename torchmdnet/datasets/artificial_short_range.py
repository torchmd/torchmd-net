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


class ShortRange(Dataset):
    def __init__(self, max_dist, size, max_z, transform=None, pre_transform=None):
        self.max_dist = max_dist
        self.size = size
        self.max_z = max_z
        # Create some npy files with random data. The dataset consists of pairs of atoms, with their positions, atomic numbers and energy
        # Positions inside a sphere of radius max_dist
        self.pos = random_vectors_in_sphere_box_muller(max_dist, 2 * size)
        self.pos = self.pos.reshape(size, 2, 3)
        # Atomic numbers
        self.z = np.random.randint(1, max_z, size=2 * size).reshape(size, 2)
        # Energy, should be a linear function of the distance, goes from 20 to 100 from max_dist to 0
        dist = np.linalg.norm(
            self.pos[:, 0, :] - self.pos[:, 1, :], axis=1
        )  # shape (size,)
        self.y = 20 + 80 * (1 - dist / max_dist)
        # Negative gradient of the energy with respect to the positions, should have the same shape as pos
        self.neg_dy = np.zeros((size, 2, 3))
        self.neg_dy[:, 0, :] = (
            -80
            / max_dist
            * (self.pos[:, 0, :] - self.pos[:, 1, :])
            / dist[:, np.newaxis]
        )
        self.neg_dy[:, 1, :] = -self.neg_dy[:, 0, :]

    def get(self, idx):
        data = Data(
            z=torch.tensor(self.z[idx], dtype=torch.long),
            pos=torch.tensor(self.pos[idx], dtype=torch.float),
            y=torch.tensor(self.y[idx], dtype=torch.float),
            neg_dy=torch.tensor(self.neg_dy[idx], dtype=torch.float),
        )
        return data

    def __len__(self):
        return self.size
