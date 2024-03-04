# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data
from torchmdnet.datasets.memdataset import MemmappedDataset
from tqdm import tqdm


class QM9q(MemmappedDataset):
    HARTREE_TO_EV = 27.211386246  #::meta private:
    BORH_TO_ANGSTROM = 0.529177  #::meta private:
    DEBYE_TO_EANG = 0.2081943  #::meta private: Debey -> e*A

    # Ion energies of elements
    ELEMENT_ENERGIES = {
        1: {0: -0.5013312007, 1: 0.0000000000},
        6: {-1: -37.8236383010, 0: -37.8038423252, 1: -37.3826165878},
        7: {-1: -54.4626446440, 0: -54.5269367415, 1: -53.9895574739},
        8: {-1: -74.9699154500, 0: -74.9812632126, 1: -74.4776884006},
        9: {-1: -99.6695561536, 0: -99.6185158728},
    }  #::meta private:

    # Select an ion with the lowest energy for each element
    INITIAL_CHARGES = {
        element: sorted(zip(charges.values(), charges.keys()))[0][1]
        for element, charges in ELEMENT_ENERGIES.items()
    }  #::meta private:

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        paths=None,
    ):
        self.name = self.__class__.__name__
        self.paths = str(paths)
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y", "neg_dy", "q", "pq", "dp"),
        )

    @property
    def raw_paths(self):
        paths = self.paths

        if os.path.isfile(paths):
            return [paths]
        if os.path.isdir(paths):
            return [
                os.path.join(paths, file_)
                for file_ in os.listdir(paths)
                if file_.endswith(".h5")
            ]

        raise RuntimeError(f"Cannot load {paths}")

    @staticmethod
    def compute_reference_energy(atomic_numbers, charge):
        atomic_numbers = np.array(atomic_numbers)
        charge = int(charge)

        charges = [QM9q.INITIAL_CHARGES[z] for z in atomic_numbers]
        energy = sum(
            QM9q.ELEMENT_ENERGIES[z][q] for z, q in zip(atomic_numbers, charges)
        )

        while sum(charges) != charge:
            dq = np.sign(charge - sum(charges))

            new_energies = []
            for i, (z, q) in enumerate(zip(atomic_numbers, charges)):
                if (q + dq) in QM9q.ELEMENT_ENERGIES[z]:
                    new_energy = (
                        energy
                        - QM9q.ELEMENT_ENERGIES[z][q]
                        + QM9q.ELEMENT_ENERGIES[z][q + dq]
                    )
                    new_energies.append((new_energy, i, q + dq))

            energy, i, q = sorted(new_energies)[0]
            charges[i] = q

        assert sum(charges) == charge

        energy = sum(
            QM9q.ELEMENT_ENERGIES[z][q] for z, q in zip(atomic_numbers, charges)
        )

        return energy * QM9q.HARTREE_TO_EV

    def sample_iter(self, mol_ids=False):
        for path in tqdm(self.raw_paths, desc="Files"):
            molecules = list(h5py.File(path).values())[0].items()

            for mol_id, mol in tqdm(molecules, desc="Molecules", leave=False):
                z = pt.tensor(mol["atomic_numbers"], dtype=pt.long)

                for conf in mol["energy"]:
                    assert mol["positions"].attrs["units"] == "Å : ångströms"
                    pos = pt.tensor(mol["positions"][conf], dtype=pt.float32)
                    assert z.shape[0] == pos.shape[0]
                    assert pos.shape[1] == 3

                    assert mol["energy"].attrs["units"] == "E_h : hartree"
                    y = (
                        pt.tensor(mol["energy"][conf][()], dtype=pt.float64)
                        * self.HARTREE_TO_EV
                    )

                    assert (
                        mol["gradient_vector"].attrs["units"]
                        == "vector : Hartree/Bohr "
                    )
                    neg_dy = (
                        -pt.tensor(mol["gradient_vector"][conf], dtype=pt.float32)
                        * self.HARTREE_TO_EV
                        / self.BORH_TO_ANGSTROM
                    )
                    assert z.shape[0] == neg_dy.shape[0]
                    assert neg_dy.shape[1] == 3

                    assert (
                        mol["electronic_charge"].attrs["units"]
                        == "n : fractional electrons"
                    )
                    pq = pt.tensor(mol["electronic_charge"][conf], dtype=pt.float32)
                    q = pq.sum().round().to(pt.long)

                    assert mol["dipole_moment"].attrs["units"] == "\\mu : Debye "
                    dp = (
                        pt.tensor(mol["dipole_moment"][conf], dtype=pt.float32)
                        * self.DEBYE_TO_EANG
                    )

                    y -= self.compute_reference_energy(z, q)

                    # Skip samples with large forces
                    if neg_dy.norm(dim=1).max() > 100:  # eV/A
                        continue

                    # Create a sample
                    args = dict(
                        z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy, q=q, pq=pq, dp=dp
                    )
                    if mol_ids:
                        args["mol_id"] = mol_id
                    data = Data(**args)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    yield data
