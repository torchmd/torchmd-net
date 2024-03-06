# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from torch_geometric.data import Data, Dataset
import numpy as np
import torch as pt
import os


class MemmappedDataset(Dataset):
    """Dataset class which stores all the molecular data in memory-mapped files.

    It supports the following attributes in the data returned by sample_iter:

        - :obj:`z`: Atomic numbers of the atoms.
        - :obj:`pos`: Positions of the atoms.
        - :obj:`y`: Energy of the conformation.
        - :obj:`neg_dy`: Forces on the atoms.
        - :obj:`total_charge`: Total charge of the conformation.
        - :obj:`partial_charges`: Partial charges of the atoms.
        - :obj:`dipole_moment`: Dipole moment of the conformation.

    The data is stored in the following files:

        - :obj:`name.idx.mmap`: Index of the first atom of each conformation.
        - :obj:`name.z.mmap`: Atomic numbers of all the atoms.
        - :obj:`name.pos.mmap`: Positions of all the atoms.
        - :obj:`name.y.mmap`: Energy of each conformation.
        - :obj:`name.neg_dy.mmap`: Forces on all the atoms.
        - :obj:`name.total_charge.mmap`: Total charge of each conformation.
        - :obj:`name.partial_charges.mmap`: Partial charges of all the atoms.
        - :obj:`name.dipole_moment.mmap`: Dipole moment of each conformation.

    Args:
        root (str): Root directory where the dataset should be stored.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset.
        properties (tuple of str, optional): The properties to include in the
            dataset. Can be any subset of :obj:`y`, :obj:`neg_dy`, :obj:`total_charge`,
            :obj:`partial_charges`, and :obj:`dipole_moment`.
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        properties=("y", "neg_dy", "total_charge", "partial_charges", "dipole_moment"),
    ):
        self.name = self.__class__.__name__
        self.properties = properties
        super().__init__(root, transform, pre_transform, pre_filter)

        fnames = self.processed_paths_dict

        self.idx_mm = np.memmap(fnames["idx"], mode="r", dtype=np.int64)
        self.z_mm = np.memmap(fnames["z"], mode="r", dtype=np.int8)
        num_all_confs = self.idx_mm.shape[0] - 1
        num_all_atoms = self.z_mm.shape[0]
        self.pos_mm = np.memmap(
            fnames["pos"], mode="r", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        if "y" in self.properties:
            self.y_mm = np.memmap(fnames["y"], mode="r", dtype=np.float64)
        if "neg_dy" in self.properties:
            self.neg_dy_mm = np.memmap(
                fnames["neg_dy"], mode="r", dtype=np.float32, shape=(num_all_atoms, 3)
            )
        if "total_charge" in self.properties:
            self.q_mm = np.memmap(fnames["total_charge"], mode="r", dtype=np.int8)
        if "partial_charges" in self.properties:
            self.pq_mm = np.memmap(fnames["partial_charges"], mode="r", dtype=np.float32)
        if "dipole_moment" in self.properties:
            self.dp_mm = np.memmap(
                fnames["dipole_moment"], mode="r", dtype=np.float32, shape=(num_all_confs, 3)
            )

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.{prop}.mmap"
            for prop in ["idx", "z", "pos"] + list(self.properties)
        ]

    @property
    def processed_paths_dict(self):
        return {
            prop: fname
            for prop, fname in zip(
                ["idx", "z", "pos"] + list(self.properties), self.processed_paths
            )
        }

    def sample_iter(self, mol_ids=False):
        raise NotImplementedError()

    def process(self):
        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]

        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")
        print(f"  Properties available: {self.properties}")

        fnames = self.processed_paths_dict

        idx_mm = np.memmap(
            fnames["idx"] + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(num_all_confs + 1,),
        )
        z_mm = np.memmap(
            fnames["z"] + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        pos_mm = np.memmap(
            fnames["pos"] + ".tmp",
            mode="w+",
            dtype=np.float32,
            shape=(num_all_atoms, 3),
        )
        if "y" in self.properties:
            y_mm = np.memmap(
                fnames["y"] + ".tmp",
                mode="w+",
                dtype=np.float64,
                shape=(num_all_confs,),
            )
        if "neg_dy" in self.properties:
            neg_dy_mm = np.memmap(
                fnames["neg_dy"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=(num_all_atoms, 3),
            )
        if "total_charge" in self.properties:
            q_mm = np.memmap(
                fnames["total_charge"] + ".tmp", mode="w+", dtype=np.int8, shape=num_all_confs
            )
        if "partial_charges" in self.properties:
            pq_mm = np.memmap(
                fnames["partial_charges"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=num_all_atoms,
            )
        if "dipole_moment" in self.properties:
            dp_mm = np.memmap(
                fnames["dipole_moment"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=(num_all_confs, 3),
            )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            if "y" in self.properties:
                y_mm[i_conf] = data.y
            if "neg_dy" in self.properties:
                neg_dy_mm[i_atom:i_next_atom] = data.neg_dy
            if "total_charge" in self.properties:
                q_mm[i_conf] = data.total_charge.to(pt.int8)
            if "partial_charges" in self.properties:
                pq_mm[i_atom:i_next_atom] = data.partial_charges
            if "dipole_moment" in self.properties:
                dp_mm[i_conf] = data.dipole_moment
            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        if "y" in self.properties:
            y_mm.flush()
        if "neg_dy" in self.properties:
            neg_dy_mm.flush()
        if "total_charge" in self.properties:
            q_mm.flush()
        if "partial_charges" in self.properties:
            pq_mm.flush()
        if "dipole_moment" in self.properties:
            dp_mm.flush()

        os.rename(idx_mm.filename, fnames["idx"])
        os.rename(z_mm.filename, fnames["z"])
        os.rename(pos_mm.filename, fnames["pos"])
        if "y" in self.properties:
            os.rename(y_mm.filename, fnames["y"])
        if "neg_dy" in self.properties:
            os.rename(neg_dy_mm.filename, fnames["neg_dy"])
        if "total_charge" in self.properties:
            os.rename(q_mm.filename, fnames["total_charge"])
        if "partial_charges" in self.properties:
            os.rename(pq_mm.filename, fnames["partial_charges"])
        if "dipole_moment" in self.properties:
            os.rename(dp_mm.filename, fnames["dipole_moment"])

    def len(self):
        return len(self.idx_mm) - 1

    def get(self, idx):
        """Gets the data object at index :obj:`idx`.

        The data object contains the following attributes:

            - :obj:`z`: Atomic numbers of the atoms.
            - :obj:`pos`: Positions of the atoms.
            - :obj:`y`: Formation energy of the molecule.
            - :obj:`neg_dy`: Forces on the atoms.
            - :obj:`total_charge`: Total charge of the molecule.
            - :obj:`partial_charges`: Partial charges of the atoms.
            - :obj:`dipole_moment`: Dipole moment of the molecule.

        Args:
            idx (int): Index of the data object.

        Returns:
            :obj:`torch_geometric.data.Data`: The data object.
        """
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms])

        props = {}
        if "y" in self.properties:
            props["y"] = pt.tensor(self.y_mm[idx]).view(1, 1)
        if "neg_dy" in self.properties:
            props["neg_dy"] = pt.tensor(self.neg_dy_mm[atoms])
        if "total_charge" in self.properties:
            props["total_charge"] = pt.tensor(self.q_mm[idx], dtype=pt.long)
        if "partial_charges" in self.properties:
            props["partial_charges"] = pt.tensor(self.pq_mm[atoms])
        if "dipole_moment" in self.properties:
            props["dipole_moment"] = pt.tensor(self.dp_mm[idx])
        return Data(z=z, pos=pos, **props)
