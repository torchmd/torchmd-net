# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, Dataset, download_url, extract_tar
from tqdm import tqdm
import warnings


class ANIBase(Dataset):
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

    HARTREE_TO_EV = 27.211386246 #::meta private:

    @property
    def raw_url(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        raise NotImplementedError

    def compute_reference_energy(self, atomic_numbers):
        atomic_numbers = np.array(atomic_numbers)
        energy = sum(self._ELEMENT_ENERGIES[z] for z in atomic_numbers)
        return energy * ANIBase.HARTREE_TO_EV

    def sample_iter(self, mol_ids=False):
        raise NotImplementedError()

    def get_atomref(self, max_z=100):
        raise NotImplementedError()

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = self.__class__.__name__
        super().__init__(root, transform, pre_transform, pre_filter)

        idx_name, z_name, pos_name, y_name, neg_dy_name = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)
        self.neg_dy_mm = (
            np.memmap(
                neg_dy_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
            )
            if os.path.getsize(neg_dy_name) > 0
            else None
        )

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx.mmap",
            f"{self.name}.z.mmap",
            f"{self.name}.pos.mmap",
            f"{self.name}.y.mmap",
            f"{self.name}.neg_dy.mmap",
        ]

    def filter_and_pre_transform(self, data):

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def process(self):
        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]
        has_neg_dy = "neg_dy" in data

        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")
        print(f"  Forces available: {has_neg_dy}")

        idx_name, z_name, pos_name, y_name, neg_dy_name = self.processed_paths
        idx_mm = np.memmap(
            idx_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs + 1,)
        )
        z_mm = np.memmap(
            z_name + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        pos_mm = np.memmap(
            pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        y_mm = np.memmap(
            y_name + ".tmp", mode="w+", dtype=np.float64, shape=(num_all_confs,)
        )
        neg_dy_mm = (
            np.memmap(
                neg_dy_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
            )
            if has_neg_dy
            else open(neg_dy_name, "w")
        )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y
            if has_neg_dy:
                neg_dy_mm[i_atom:i_next_atom] = data.neg_dy

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        if has_neg_dy:
            neg_dy_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        if has_neg_dy:
            os.rename(neg_dy_mm.filename, neg_dy_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):
        """Get a single sample from the dataset.

        Data object contains the following attributes by default:

        - :obj:`z` (:class:`torch.LongTensor`): Atomic numbers of shape :obj:`[num_nodes]`.
        - :obj:`pos` (:class:`torch.FloatTensor`): Atomic positions of shape :obj:`[num_nodes, 3]`.
        - :obj:`y` (:class:`torch.FloatTensor`): Energies of shape :obj:`[1, 1]`.
        - :obj:`neg_dy` (:class:`torch.FloatTensor`, *optional*): Negative gradients of shape :obj:`[num_nodes, 3]`.

        Args:
            idx (int): Index of the sample.

        Returns:
            :class:`torch_geometric.data.Data`: The data object.
        """
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        y -= self.compute_reference_energy(z)

        if self.neg_dy_mm is None:
            return Data(z=z, pos=pos, y=y)
        else:
            neg_dy = pt.tensor(self.neg_dy_mm[atoms], dtype=pt.float32)
            return Data(z=z, pos=pos, y=y, neg_dy=neg_dy)


class ANI1(ANIBase):
    __doc__ = ANIBase.__doc__
    # Avoid sphinx from documenting this
    ELEMENT_ENERGIES = {
        1: -0.500607632585,
        6: -37.8302333826,
        7: -54.5680045287,
        8: -75.0362229210,
    } #::meta private:

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

    def get_atomref(self, max_z=100):
        """Atomic energy reference values for the :py:mod:`torchmdnet.priors.Atomref` prior.
        """
        refs = pt.zeros(max_z)
        refs[1] = -0.500607632585 * self.HARTREE_TO_EV  # H
        refs[6] = -37.8302333826 * self.HARTREE_TO_EV  # C
        refs[7] = -54.5680045287 * self.HARTREE_TO_EV  # N
        refs[8] = -75.0362229210 * self.HARTREE_TO_EV  # O

        return refs.view(-1, 1)

    # Circumvent https://github.com/pyg-team/pytorch_geometric/issues/4567
    # TODO remove when fixed
    def process(self):
        super().process()


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

    def get_atomref(self, max_z=100):
        """Atomic energy reference values for the :py:mod:`torchmdnet.priors.Atomref` prior.
        """
        warnings.warn("Atomic references from the ANI-1 dataset are used!")

        refs = pt.zeros(max_z)
        refs[1] = -0.500607632585 * self.HARTREE_TO_EV  # H
        refs[6] = -37.8302333826 * self.HARTREE_TO_EV  # C
        refs[7] = -54.5680045287 * self.HARTREE_TO_EV  # N
        refs[8] = -75.0362229210 * self.HARTREE_TO_EV  # O

        return refs.view(-1, 1)


class ANI1X(ANI1XBase):
    __doc__ = ANIBase.__doc__
    ELEMENT_ENERGIES = {
        1: -0.500607632585,
        6: -37.8302333826,
        7: -54.5680045287,
        8: -75.0362229210,
    }
    """

    :meta private:
    """
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

    # Circumvent https://github.com/pyg-team/pytorch_geometric/issues/4567
    # TODO remove when fixed
    def download(self):
        super().download()

    # Circumvent https://github.com/pyg-team/pytorch_geometric/issues/4567
    # TODO remove when fixed
    def process(self):
        super().process()


class ANI1CCX(ANI1XBase):
    __doc__ = ANIBase.__doc__
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

    # Circumvent https://github.com/pyg-team/pytorch_geometric/issues/4567
    # TODO remove when fixed
    def download(self):
        super().download()

    # Circumvent https://github.com/pyg-team/pytorch_geometric/issues/4567
    # TODO remove when fixed
    def process(self):
        super().process()
