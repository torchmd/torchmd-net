# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, download_url, extract_tar
from torchmdnet.datasets.memdataset import MemmappedDataset
from tqdm import tqdm
import warnings


class ANIBase(MemmappedDataset):
    """ANI Dataset Classes

    A foundational dataset class for handling the ANI datasets. ANI (ANAKIN-ME or Accurate NeurAl networK engINe for Molecular Energies)
    is a deep learning method trained on quantum mechanical DFT calculations to predict accurate and transferable potentials for organic molecules.

    Key features of ANI:

    - Utilizes a modified version of the Behler and Parrinello symmetry functions to construct single-atom atomic environment vectors (AEV) for molecular representation.
    - AEVs enable the training of neural networks over both configurational and conformational space.
    - The ANI-1 potential was trained on a subset of the GDB databases with up to 8 heavy atoms.
    - ANI-1x and ANI-1ccx datasets provide diverse quantum mechanical properties for organic molecules:
       -  ANI-1x contains multiple QM properties from 5M density functional theory calculations.
       -  ANI-1ccx contains 500k data points obtained with an accurate CCSD(T)/CBS extrapolation.
    - Properties include energies, atomic forces, multipole moments, atomic charges, and more for the chemical elements C, H, N, and O.
    - Developed through active learning, an automated data diversification process.

    References:

    - Smith, J. S., Isayev, O., & Roitberg, A. E. (2017). ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. Chemical Science, 8(4), 3192-3203.
    - Smith, J. S., Zubatyuk, R., Nebgen, B., Lubbers, N., Barros, K., Roitberg, A. E., Isayev, O., & Tretiak, S. (2020). The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional theory properties for molecules. Scientific Data, 7, Article 134.
    """

    HARTREE_TO_EV = 27.211386246  #::meta private:

    @property
    def raw_url(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        raise NotImplementedError

    def get_atomref(self, max_z=100):
        """Atomic energy reference values for the :py:mod:`torchmdnet.priors.Atomref` prior.

        Args:
            max_z (int): Maximum atomic number

        Returns:
            torch.Tensor: Atomic energy reference values for each element in the dataset.
        """
        refs = pt.zeros(max_z)
        for key, val in self._ELEMENT_ENERGIES.items():
            refs[key] = val * self.HARTREE_TO_EV

        return refs.view(-1, 1)

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        properties=("y", "neg_dy"),
    ):
        self.name = self.__class__.__name__
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=properties,
        )

    def filter_and_pre_transform(self, data):
        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data


class ANI1(ANIBase):
    __doc__ = ANIBase.__doc__
    _ELEMENT_ENERGIES = {
        1: -0.500607632585,
        6: -37.8302333826,
        7: -54.5680045287,
        8: -75.0362229210,
    }

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = self.__class__.__name__
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y",),
        )

    @property
    def raw_url(self):
        return "https://ndownloader.figshare.com/files/9057631"

    @property
    def raw_file_names(self):
        return [
            os.path.join("ANI-1_release", f"ani_gdb_s{i:02d}.h5") for i in range(1, 9)
        ]

    def download(self):
        archive = download_url(self.raw_url, self.raw_dir)
        extract_tar(archive, self.raw_dir)
        os.remove(archive)

    def sample_iter(self, mol_ids=False):
        atomic_numbers = {b"H": 1, b"C": 6, b"N": 7, b"O": 8}

        for path in tqdm(self.raw_paths, desc="Files"):
            molecules = list(h5py.File(path).values())[0].items()

            for mol_id, mol in tqdm(molecules, desc="Molecules", leave=False):
                z = pt.tensor(
                    [atomic_numbers[atom] for atom in mol["species"]], dtype=pt.long
                )
                all_pos = pt.tensor(mol["coordinates"][:], dtype=pt.float32)
                all_y = pt.tensor(
                    mol["energies"][:] * self.HARTREE_TO_EV, dtype=pt.float64
                )

                assert all_pos.shape[0] == all_y.shape[0]
                assert all_pos.shape[1] == z.shape[0]
                assert all_pos.shape[2] == 3

                for pos, y in zip(all_pos, all_y):
                    # Create a sample
                    args = dict(z=z, pos=pos, y=y.view(1, 1))
                    if mol_ids:
                        args["mol_id"] = mol_id
                    data = Data(**args)

                    if data := self.filter_and_pre_transform(data):
                        yield data


class ANI1XBase(ANIBase):
    @property
    def raw_url(self):
        return "https://figshare.com/ndownloader/files/18112775"

    @property
    def raw_file_names(self):
        return "ani1x-release.h5"

    def download(self):
        file = download_url(self.raw_url, self.raw_dir)
        assert len(self.raw_paths) == 1
        os.rename(file, self.raw_paths[0])


class ANI1X(ANI1XBase):
    __doc__ = ANIBase.__doc__
    _ELEMENT_ENERGIES = {
        1: -0.600952980000,
        6: -38.08316124000,
        7: -54.70775770000,
        8: -75.19446356000,
    }

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1

        with h5py.File(self.raw_paths[0]) as h5:
            for mol_id, mol in tqdm(h5.items(), desc="Molecules"):
                z = pt.tensor(mol["atomic_numbers"][:], dtype=pt.long)
                all_pos = pt.tensor(mol["coordinates"][:], dtype=pt.float32)
                all_y = pt.tensor(
                    mol["wb97x_dz.energy"][:] * self.HARTREE_TO_EV, dtype=pt.float64
                )
                all_neg_dy = pt.tensor(
                    mol["wb97x_dz.forces"][:] * self.HARTREE_TO_EV, dtype=pt.float32
                )

                assert all_pos.shape[0] == all_y.shape[0]
                assert all_pos.shape[1] == z.shape[0]
                assert all_pos.shape[2] == 3

                assert all_neg_dy.shape[0] == all_y.shape[0]
                assert all_neg_dy.shape[1] == z.shape[0]
                assert all_neg_dy.shape[2] == 3

                for pos, y, neg_dy in zip(all_pos, all_y, all_neg_dy):
                    if y.isnan() or neg_dy.isnan().any():
                        continue

                    # Create a sample
                    args = dict(z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy)
                    if mol_ids:
                        args["mol_id"] = mol_id
                    data = Data(**args)

                    if data := self.filter_and_pre_transform(data):
                        yield data


class ANI1CCX(ANI1XBase):
    __doc__ = ANIBase.__doc__
    _ELEMENT_ENERGIES = {
        1: -0.5991501324919538,
        6: -38.03750806057356,
        7: -54.67448347695333,
        8: -75.16043537275567,
    }

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = self.__class__.__name__
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y",),
        )

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1

        with h5py.File(self.raw_paths[0]) as h5:
            for mol_id, mol in tqdm(h5.items(), desc="Molecules"):
                z = pt.tensor(mol["atomic_numbers"][:], dtype=pt.long)
                all_pos = pt.tensor(mol["coordinates"][:], dtype=pt.float32)
                all_y = pt.tensor(
                    mol["ccsd(t)_cbs.energy"][:] * self.HARTREE_TO_EV, dtype=pt.float64
                )

                assert all_pos.shape[0] == all_y.shape[0]
                assert all_pos.shape[1] == z.shape[0]
                assert all_pos.shape[2] == 3

                for pos, y in zip(all_pos, all_y):
                    if y.isnan():
                        continue

                    # Create a sample
                    args = dict(z=z, pos=pos, y=y.view(1, 1))
                    if mol_ids:
                        args["mol_id"] = mol_id
                    data = Data(**args)

                    if data := self.filter_and_pre_transform(data):
                        yield data


class ANI2X(ANIBase):
    __doc__ = ANIBase.__doc__

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

    @property
    def raw_url(self):
        return "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz"

    @property
    def raw_file_names(self):
        return [os.path.join("final_h5", "ANI-2x-wB97X-631Gd.h5")]

    def download(self):
        archive = download_url(self.raw_url, self.raw_dir)
        extract_tar(archive, self.raw_dir)
        os.remove(archive)

    def sample_iter(self, mol_ids=False):
        """
        In [15]: list(molecules)
        Out[15]:
        [('coordinates', <HDF5 dataset "coordinates": shape (5706, 2, 3), type "<f4">),
        ('energies', <HDF5 dataset "energies": shape (5706,), type "<f8">),
        ('forces', <HDF5 dataset "forces": shape (5706, 2, 3), type "<f8">),
        ('species', <HDF5 dataset "species": shape (5706, 2), type "<i8">)]
        """
        assert len(self.raw_paths) == 1
        with h5py.File(self.raw_paths[0]) as h5data:
            for key, data in tqdm(h5data.items(), desc="Molecule Group", leave=False):
                all_z = pt.tensor(data["species"][:], dtype=pt.long)
                all_pos = pt.tensor(data["coordinates"][:], dtype=pt.float32)
                all_y = pt.tensor(
                    data["energies"][:] * self.HARTREE_TO_EV, dtype=pt.float64
                )
                all_neg_dy = pt.tensor(
                    data["forces"][:] * self.HARTREE_TO_EV, dtype=pt.float32
                )
                n_mols = all_pos.shape[0]
                n_atoms = all_pos.shape[1]

                assert all_y.shape[0] == n_mols
                assert all_z.shape == (n_mols, n_atoms)
                assert all_pos.shape == (n_mols, n_atoms, 3)
                assert all_neg_dy.shape == (n_mols, n_atoms, 3)

                for i, (pos, y, z, neg_dy) in enumerate(
                    zip(all_pos, all_y, all_z, all_neg_dy)
                ):
                    # Create a sample
                    args = dict(z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy)
                    if mol_ids:
                        args["mol_id"] = f"{key}_{i}"
                    data = Data(**args)

                    if data := self.filter_and_pre_transform(data):
                        yield data
