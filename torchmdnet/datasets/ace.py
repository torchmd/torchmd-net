# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import hashlib
import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class Ace(Dataset):
    """The ACE dataset.

    This dataset is sourced from HDF5 files.

    Mandatory HDF5 file attributes:

    - `layout`: Must be set to `Ace`.
    - `layout_version`: Can be `1.0` or `2.0`.
    - `name`: Name of the dataset.

    For `layout_version` 1.0:

    - Files can contain multiple molecule groups directly under the root.
    - Each molecule group contains:

      - `atomic_numbers`: Atomic numbers of the atoms.
      - `formal_charges`: Formal charges of the atoms. The sum is the molecule's total charge. Units: electron charges.
      - `conformations` subgroup: This subgroup has individual conformation groups, each with datasets for different properties of the conformation.

    For `layout_version` 2.0:

    - Files contain a single root group (e.g., a 'master molecule group').
    - Within this root group, there can be multiple molecule groups.
    - Each molecule group contains:

      - `atomic_numbers`: Atomic numbers of the atoms.
      - `formal_charges`: Formal charges of the atoms.
      - Datasets for multiple conformations directly, without individual conformation groups.


    Each conformation group (version 1.0) or molecule group (version 2.0) should have the following datasets:

    - `positions`: Atomic positions. Units: Angstrom.
    - `forces`: Forces on the atoms. Units: eV/Å.
    - `partial_charges`: Atomic partial charges. Units: electron charges.
    - `dipole_moment` (version 1.0) or `dipole_moments` (version 2.0): Dipole moment (a vector of three components). Units: e*Å.
    - `formation_energy` (version 1.0) or `formation_energies` (version 2.0): Formation energy. Units: eV.

    Each dataset should also have an `units` attribute specifying its units (i.e., `Å`, `eV`, `e*Å`).

    Note that version 2.0 is more efficient than 1.0.

    Args:
        root (string, optional): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset.
        paths (string or list): Path to the HDF5 files or directory containing the HDF5 files.
        max_gradient (float, optional): Maximum gradient norm. Samples with larger gradients are discarded.
        subsample_molecules (int, optional): Subsample molecules. Only every `subsample_molecules` molecule is used.

    Examples::
        >>> import numpy as np
        >>> from torchmdnet.datasets import Ace
        >>> import h5py
        >>> # Version 1.0 example
        >>> with h5py.File("molecule.h5", 'w') as f:
        ...     f.attrs["layout"] = "Ace"
        ...     f.attrs["layout_version"] = "1.0"
        ...     f.attrs["name"] = "sample_molecule_data"
        ...     for m in range(3):  # Three molecules
        ...         mol = f.create_group(f"mol_{m+1}")
        ...         mol["atomic_numbers"] = [1, 6, 8]  # H, C, O
        ...         mol["formal_charges"] = [0, 0, 0]  # Neutral charges
        ...         confs = mol.create_group("conformations")
        ...         for i in range(2):  # Two conformations
        ...             conf = confs.create_group(f"conf_{i+1}")
        ...             conf["positions"] = np.random.random((3, 3))
        ...             conf["positions"].attrs["units"] = "Å"
        ...             conf["formation_energy"] = np.random.random()
        ...             conf["formation_energy"].attrs["units"] = "eV"
        ...             conf["forces"] = np.random.random((3, 3))
        ...             conf["forces"].attrs["units"] = "eV/Å"
        ...             conf["partial_charges"] = np.random.random(3)
        ...             conf["partial_charges"].attrs["units"] = "e"
        ...             conf["dipole_moment"] = np.random.random(3)
        ...             conf["dipole_moment"].attrs["units"] = "e*Å"
        >>> dataset = Ace(root=".", paths="molecule.h5")
        >>> len(dataset)
        6
        >>> dataset = Ace(root=".", paths=["molecule.h5", "molecule.h5"])
        >>> len(dataset)
        12
        >>> # Version 2.0 example
        >>> with h5py.File("molecule_v2.h5", 'w') as f:
        ...     f.attrs["layout"] = "Ace"
        ...     f.attrs["layout_version"] = "2.0"
        ...     f.attrs["name"] = "sample_molecule_data_v2"
        ...     master_mol_group = f.create_group("master_molecule_group")
        ...     for m in range(3):  # Three molecules
        ...         mol = master_mol_group.create_group(f"mol_{m+1}")
        ...         mol["atomic_numbers"] = [1, 6, 8]  # H, C, O
        ...         mol["formal_charges"] = [0, 0, 0]  # Neutral charges
        ...         mol["positions"] = np.random.random((2, 3, 3))  # Two conformations
        ...         mol["positions"].attrs["units"] = "Å"
        ...         mol["formation_energies"] = np.random.random(2)
        ...         mol["formation_energies"].attrs["units"] = "eV"
        ...         mol["forces"] = np.random.random((2, 3, 3))
        ...         mol["forces"].attrs["units"] = "eV/Å"
        ...         mol["partial_charges"] = np.random.random((2, 3))
        ...         mol["partial_charges"].attrs["units"] = "e"
        ...         mol["dipole_moment"] = np.random.random((2, 3))
        ...         mol["dipole_moment"].attrs["units"] = "e*Å"
        >>> dataset_v2 = Ace(root=".", paths="molecule_v2.h5")
        >>> len(dataset_v2)
        6
        >>> dataset_v2 = Ace(root=".", paths=["molecule_v2.h5", "molecule_v2.h5"])
        >>> len(dataset_v2)
        12
    """

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        paths=None,
        max_gradient=None,
        subsample_molecules=1,
    ):
        assert isinstance(paths, (str, list))

        arg_hash = f"{paths}{max_gradient}{subsample_molecules}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.paths = paths
        self.max_gradient = max_gradient
        self.subsample_molecules = int(subsample_molecules)
        super().__init__(root, transform, pre_transform, pre_filter)

        (
            idx_name,
            z_name,
            pos_name,
            y_name,
            neg_dy_name,
            q_name,
            pq_name,
            dp_name,
        ) = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)
        self.neg_dy_mm = np.memmap(
            neg_dy_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.q_mm = np.memmap(q_name, mode="r", dtype=np.int8)
        self.pq_mm = np.memmap(pq_name, mode="r", dtype=np.float32)
        self.dp_mm = np.memmap(
            dp_name, mode="r", dtype=np.float32, shape=(self.y_mm.shape[0], 3)
        )

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1

    @property
    def raw_paths(self):

        paths_init = self.paths if isinstance(self.paths, list) else [self.paths]
        paths = []
        for path in paths_init:

            if os.path.isfile(path):
                paths.append(path)
                continue

            if os.path.isdir(path):
                for file_ in os.listdir(path):
                    if file_.endswith(".h5"):
                        paths.append(os.path.join(path, file_))
                continue

            raise RuntimeError(f"{path} is neither a directory nor a file")

        return paths

    @staticmethod
    def _load_confs_1_0(mol, n_atoms):

        for conf in mol["conformations"].values():

            # Skip failed calculations
            if "formation_energy" not in conf:
                continue

            assert conf["positions"].attrs["units"] == "Å"
            pos = pt.tensor(conf["positions"][...], dtype=pt.float32)
            assert pos.shape == (n_atoms, 3)

            assert conf["formation_energy"].attrs["units"] == "eV"
            y = pt.tensor(conf["formation_energy"][()], dtype=pt.float64)
            assert y.shape == ()

            assert conf["forces"].attrs["units"] == "eV/Å"
            neg_dy = pt.tensor(conf["forces"][...], dtype=pt.float32)
            assert neg_dy.shape == pos.shape

            assert conf["partial_charges"].attrs["units"] == "e"
            pq = pt.tensor(conf["partial_charges"][:], dtype=pt.float32)
            assert pq.shape == (n_atoms,)

            assert conf["dipole_moment"].attrs["units"] == "e*Å"
            dp = pt.tensor(conf["dipole_moment"][:], dtype=pt.float32)
            assert dp.shape == (3,)

            yield pos, y, neg_dy, pq, dp

    @staticmethod
    def _load_confs_2_0(mol, n_atoms):

        assert mol["positions"].attrs["units"] == "Å"
        all_pos = pt.tensor(mol["positions"][...], dtype=pt.float32)
        n_confs = all_pos.shape[0]
        assert all_pos.shape == (n_confs, n_atoms, 3)

        assert mol["formation_energies"].attrs["units"] == "eV"
        all_y = pt.tensor(mol["formation_energies"][:], dtype=pt.float64)
        assert all_y.shape == (n_confs,)

        assert mol["forces"].attrs["units"] == "eV/Å"
        all_neg_dy = pt.tensor(mol["forces"][...], dtype=pt.float32)
        assert all_neg_dy.shape == all_pos.shape

        assert mol["partial_charges"].attrs["units"] == "e"
        all_pq = pt.tensor(mol["partial_charges"][...], dtype=pt.float32)
        assert all_pq.shape == (n_confs, n_atoms)

        assert mol["dipole_moments"].attrs["units"] == "e*Å"
        all_dp = pt.tensor(mol["dipole_moments"][...], dtype=pt.float32)
        assert all_dp.shape == (n_confs, 3)

        for pos, y, neg_dy, pq, dp in zip(all_pos, all_y, all_neg_dy, all_pq, all_dp):

            # Skip failed calculations
            if y.isnan():
                continue

            yield pos, y, neg_dy, pq, dp

    def sample_iter(self, mol_ids=False):

        assert self.subsample_molecules > 0

        for path in tqdm(self.raw_paths, desc="Files"):

            h5 = h5py.File(path)
            assert h5.attrs["layout"] == "Ace"
            version = h5.attrs["layout_version"]

            mols = None
            load_confs = None
            if version == "1.0":
                assert "name" in h5.attrs
                mols = h5.items()
                load_confs = self._load_confs_1_0
            elif version == "2.0":
                assert len(h5.keys()) == 1
                mols = list(h5.values())[0].items()
                load_confs = self._load_confs_2_0
            else:
                raise RuntimeError(f"Unsupported layout version: {version}")

            # Iterate over the molecules
            for i_mol, (mol_id, mol) in tqdm(
                enumerate(mols),
                desc="Molecules",
                total=len(mols),
                leave=False,
            ):

                # Subsample molecules
                if i_mol % self.subsample_molecules != 0:
                    continue

                z = pt.tensor(mol["atomic_numbers"], dtype=pt.long)
                fq = pt.tensor(mol["formal_charges"], dtype=pt.long)
                q = fq.sum()

                for i_conf, (pos, y, neg_dy, pq, dp) in enumerate(load_confs(mol, n_atoms=len(z))):

                    # Skip samples with large forces
                    if self.max_gradient:
                        if neg_dy.norm(dim=1).max() > float(self.max_gradient):
                            continue

                    # Create a sample
                    args = dict(
                        z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy, q=q, pq=pq, dp=dp
                    )
                    if mol_ids:
                        args["mol_id"] = mol_id
                        args["i_conf"] = i_conf
                    data = Data(**args)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    yield data

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx.mmap",
            f"{self.name}.z.mmap",
            f"{self.name}.pos.mmap",
            f"{self.name}.y.mmap",
            f"{self.name}.neg_dy.mmap",
            f"{self.name}.q.mmap",
            f"{self.name}.pq.mmap",
            f"{self.name}.dp.mmap",
        ]

    def process(self):

        print("Arguments")
        print(f"  max_gradient: {self.max_gradient} eV/A")
        print(f"  subsample_molecules: {self.subsample_molecules}\n")

        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]

        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")

        (
            idx_name,
            z_name,
            pos_name,
            y_name,
            neg_dy_name,
            q_name,
            pq_name,
            dp_name,
        ) = self.processed_paths
        idx_mm = np.memmap(
            idx_name + ".tmp", mode="w+", dtype=np.int64, shape=num_all_confs + 1
        )
        z_mm = np.memmap(z_name + ".tmp", mode="w+", dtype=np.int8, shape=num_all_atoms)
        pos_mm = np.memmap(
            pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        y_mm = np.memmap(
            y_name + ".tmp", mode="w+", dtype=np.float64, shape=num_all_confs
        )
        neg_dy_mm = np.memmap(
            neg_dy_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        q_mm = np.memmap(q_name + ".tmp", mode="w+", dtype=np.int8, shape=num_all_confs)
        pq_mm = np.memmap(
            pq_name + ".tmp", mode="w+", dtype=np.float32, shape=num_all_atoms
        )
        dp_mm = np.memmap(
            dp_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_confs, 3)
        )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y
            neg_dy_mm[i_atom:i_next_atom] = data.neg_dy
            q_mm[i_conf] = data.q.to(pt.int8)
            pq_mm[i_atom:i_next_atom] = data.pq
            dp_mm[i_conf] = data.dp

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        neg_dy_mm.flush()
        q_mm.flush()
        pq_mm.flush()
        dp_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(neg_dy_mm.filename, neg_dy_name)
        os.rename(q_mm.filename, q_name)
        os.rename(pq_mm.filename, pq_name)
        os.rename(dp_mm.filename, dp_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):
        """Gets the data object at index :obj:`idx`.

        The data object contains the following attributes:

            - :obj:`z`: Atomic numbers of the atoms.
            - :obj:`pos`: Positions of the atoms.
            - :obj:`y`: Formation energy of the molecule.
            - :obj:`neg_dy`: Forces on the atoms.
            - :obj:`q`: Total charge of the molecule.
            - :obj:`pq`: Partial charges of the atoms.
            - :obj:`dp`: Dipole moment of the molecule.

        Args:
            idx (int): Index of the data object.

        Returns:
            :obj:`torch_geometric.data.Data`: The data object.
        """
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        neg_dy = pt.tensor(self.neg_dy_mm[atoms], dtype=pt.float32)
        q = pt.tensor(self.q_mm[idx], dtype=pt.long)
        pq = pt.tensor(self.pq_mm[atoms], dtype=pt.float32)
        dp = pt.tensor(self.dp_mm[idx], dtype=pt.float32)

        return Data(z=z, pos=pos, y=y, neg_dy=neg_dy, q=q, pq=pq, dp=dp)
