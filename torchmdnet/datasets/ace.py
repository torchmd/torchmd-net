# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import hashlib
import h5py
import os
import torch as pt
from torchmdnet.datasets.memdataset import MemmappedDataset
from torch_geometric.data import Data
from tqdm import tqdm


class Ace(MemmappedDataset):
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

        props = ["y", "neg_dy", "q", "pq", "dp"]
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=tuple(props),
        )

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

        for i_path, path in tqdm(enumerate(self.raw_paths), desc="Files"):
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

                for i_conf, (pos, y, neg_dy, pq, dp) in enumerate(
                    load_confs(mol, n_atoms=len(z))
                ):
                    # Skip samples with large forces
                    if self.max_gradient:
                        if neg_dy.norm(dim=1).max() > float(self.max_gradient):
                            continue

                    # Create a sample
                    args = dict(
                        z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy, q=q, pq=pq, dp=dp
                    )
                    if mol_ids:
                        args["i_path"] = i_path
                        args["mol_id"] = mol_id
                        args["i_conf"] = i_conf

                    data = Data(**args)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    yield data
