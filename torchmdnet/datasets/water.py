from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
import os
import requests
import zipfile
import re


def create_numpy_arrays(file_path):
    with open(file_path, "r") as file:
        num_atoms = int(file.readline().strip())
        file.seek(0)
        num_conformations = sum(1 for line in file if line.strip().isdigit())
        file.seek(0)
        energies = np.zeros((num_conformations, 1))
        forces = np.zeros((num_conformations, num_atoms, 3))
        positions = np.zeros((num_conformations, num_atoms, 3))
        atomic_numbers = np.zeros((num_conformations, num_atoms, 1), dtype=int)
        box_vectors = np.zeros((num_conformations, 9))
        # Extracting TotEnergy, pbc, and Lattice
        for i in range(num_conformations):
            _ = file.readline()
            properties_line = file.readline()
            tot_energy_match = re.search(r"TotEnergy=(-?\d+\.\d+)", properties_line)
            pbc_match = re.search(r'pbc="([T|F] [T|F] [T|F])"', properties_line)
            lattice_match = re.search(r'Lattice="([-?\d+.\d+\s]+)"', properties_line)
            energies[i] = float(tot_energy_match.group(1)) if tot_energy_match else None
            pbc = [s == "T" for s in pbc_match.group(1).split()] if pbc_match else None
            assert pbc == [True, True, True] or pbc == [False, False, False]
            box_vectors[i] = (
                [float(x) for x in lattice_match.group(1).split()]
                if lattice_match
                else None
            )
            for j in range(num_atoms):
                atom_line = file.readline().strip().split()
                positions[i, j] = [float(x) for x in atom_line[1:4]]
                forces[i, j] = [float(x) for x in atom_line[4:7]]
                atomic_numbers[i, j] = int(atom_line[7])
    return energies, forces, positions, atomic_numbers, box_vectors


class WaterBox(InMemoryDataset):
    """WaterBox dataset from [1]_.

    The dataset consists of 1593 water molecules in a cubic box with periodic boundary conditions.
    The molecules are sampled from a molecular dynamics simulation of liquid water.

    Each sample in the dataset contains the following properties:

    - z (LongTensor): Atomic numbers of the atoms in the molecule.
    - pos (FloatTensor): Positions of the atoms in the molecule.
    - y (FloatTensor): Total energy of the molecule.
    - neg_dy (FloatTensor): Negative of the forces on the atoms in the molecule.
    - box (FloatTensor): Box vectors of the simulation cell.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.

    References
    ----------
    [1] Ab initio thermodynamics of liquid and solid water. Bingqing et. al. https://arxiv.org/abs/1811.08630
    """

    url = "https://archive.materialscloud.org/record/file?record_id=71&filename=training-set.zip"

    def __init__(self, root, transform=None, pre_transform=None):
        super(WaterBox, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset_1593.xyz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        r = requests.get(self.url)
        if r.status_code != 200:
            raise Exception(
                f"Failed to download file from {self.url}. Status code: {r.status_code}"
            )
        zip_path = os.path.join(self.raw_dir, "training-set.zip")
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

    def process(self):
        dataset_xyz_path = os.path.join(
            self.raw_dir, "training-set", "dataset_1593.xyz"
        )
        energies, forces, positions, atomic_numbers, box_vectors = create_numpy_arrays(
            dataset_xyz_path
        )

        data_list = []
        for i in range(len(energies)):
            z = torch.tensor(atomic_numbers[i], dtype=torch.long).view(-1)
            pos = torch.tensor(positions[i], dtype=torch.float)
            y = torch.tensor(energies[i], dtype=torch.float)
            neg_dy = torch.tensor(forces[i], dtype=torch.float)
            box = torch.tensor(box_vectors[i], dtype=torch.float).view(1, 3, 3)
            data_list.append(Data(z=z, pos=pos, y=y, neg_dy=neg_dy, box=box))

        self.save(data_list, self.processed_paths[0])
