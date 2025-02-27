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
        - :obj:`q`: Total charge of the conformation.
        - :obj:`pq`: Partial charges of the atoms.
        - :obj:`dp`: Dipole moment of the conformation.

    The data is stored in the following files:

        - :obj:`name.idx.mmap`: Index of the first atom of each conformation.
        - :obj:`name.z.mmap`: Atomic numbers of all the atoms.
        - :obj:`name.pos.mmap`: Positions of all the atoms.
        - :obj:`name.y.mmap`: Energy of each conformation.
        - :obj:`name.neg_dy.mmap`: Forces on all the atoms.
        - :obj:`name.q.mmap`: Total charge of each conformation.
        - :obj:`name.pq.mmap`: Partial charges of all the atoms.
        - :obj:`name.dp.mmap`: Dipole moment of each conformation.

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
            dataset. Can be any subset of :obj:`y`, :obj:`neg_dy`, :obj:`q`,
            :obj:`pq`, and :obj:`dp`.
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        properties=("y", "neg_dy", "q", "pq", "dp"),
    ):
        if not hasattr(self, "name"):
            self.name = self.__class__.__name__
        self.properties = properties
        super().__init__(root, transform, pre_transform, pre_filter)

        fnames = self.processed_paths_dict

        self.mmaps = {}
        self.mmaps["idx"] = np.memmap(fnames["idx"], mode="r", dtype=np.int64)
        self.mmaps["z"] = np.memmap(fnames["z"], mode="r", dtype=np.int8)
        num_all_confs = self.mmaps["idx"].shape[0] - 1
        num_all_atoms = self.mmaps["z"].shape[0]
        self.mmaps["pos"] = np.memmap(
            fnames["pos"], mode="r", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        if "y" in self.properties:
            self.mmaps["y"] = np.memmap(fnames["y"], mode="r", dtype=np.float64)
        if "neg_dy" in self.properties:
            self.mmaps["neg_dy"] = np.memmap(
                fnames["neg_dy"], mode="r", dtype=np.float32, shape=(num_all_atoms, 3)
            )
        if "q" in self.properties:
            self.mmaps["q"] = np.memmap(fnames["q"], mode="r", dtype=np.int8)
        if "pq" in self.properties:
            self.mmaps["pq"] = np.memmap(fnames["pq"], mode="r", dtype=np.float32)
        if "dp" in self.properties:
            self.mmaps["dp"] = np.memmap(
                fnames["dp"], mode="r", dtype=np.float32, shape=(num_all_confs, 3)
            )

        assert self.mmaps["idx"][0] == 0
        assert self.mmaps["idx"][-1] == len(self.mmaps["z"])
        assert len(self.mmaps["idx"]) == len(self.mmaps["y"]) + 1

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
        import gc

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

        mmaps = {}
        mmaps["idx"] = np.memmap(
            fnames["idx"] + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(num_all_confs + 1,),
        )
        mmaps["z"] = np.memmap(
            fnames["z"] + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        mmaps["pos"] = np.memmap(
            fnames["pos"] + ".tmp",
            mode="w+",
            dtype=np.float32,
            shape=(num_all_atoms, 3),
        )
        if "y" in self.properties:
            mmaps["y"] = np.memmap(
                fnames["y"] + ".tmp",
                mode="w+",
                dtype=np.float64,
                shape=(num_all_confs,),
            )
        if "neg_dy" in self.properties:
            mmaps["neg_dy"] = np.memmap(
                fnames["neg_dy"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=(num_all_atoms, 3),
            )
        if "q" in self.properties:
            mmaps["q"] = np.memmap(
                fnames["q"] + ".tmp", mode="w+", dtype=np.int8, shape=num_all_confs
            )
        if "pq" in self.properties:
            mmaps["pq"] = np.memmap(
                fnames["pq"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=num_all_atoms,
            )
        if "dp" in self.properties:
            mmaps["dp"] = np.memmap(
                fnames["dp"] + ".tmp",
                mode="w+",
                dtype=np.float32,
                shape=(num_all_confs, 3),
            )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            mmaps["idx"][i_conf] = i_atom
            mmaps["z"][i_atom:i_next_atom] = data.z.to(pt.int8)
            mmaps["pos"][i_atom:i_next_atom] = data.pos
            if "y" in self.properties:
                mmaps["y"][i_conf] = data.y
            if "neg_dy" in self.properties:
                mmaps["neg_dy"][i_atom:i_next_atom] = data.neg_dy
            if "q" in self.properties:
                mmaps["q"][i_conf] = data.q.to(pt.int8)
            if "pq" in self.properties:
                mmaps["pq"][i_atom:i_next_atom] = data.pq
            if "dp" in self.properties:
                mmaps["dp"][i_conf] = data.dp
            i_atom = i_next_atom

        mmaps["idx"][-1] = num_all_atoms
        assert i_atom == num_all_atoms

        for prop in list(mmaps.keys()):
            mmaps[prop].flush()
            del mmaps[prop]

        # Force garbage collection to ensure files are closed before renaming
        gc.collect()

        os.rename(fnames["idx"] + ".tmp", fnames["idx"])
        os.rename(fnames["z"] + ".tmp", fnames["z"])
        os.rename(fnames["pos"] + ".tmp", fnames["pos"])
        if "y" in self.properties:
            os.rename(fnames["y"] + ".tmp", fnames["y"])
        if "neg_dy" in self.properties:
            os.rename(fnames["neg_dy"] + ".tmp", fnames["neg_dy"])
        if "q" in self.properties:
            os.rename(fnames["q"] + ".tmp", fnames["q"])
        if "pq" in self.properties:
            os.rename(fnames["pq"] + ".tmp", fnames["pq"])
        if "dp" in self.properties:
            os.rename(fnames["dp"] + ".tmp", fnames["dp"])

    def len(self):
        return len(self.mmaps["idx"]) - 1

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
        atoms = slice(self.mmaps["idx"][idx], self.mmaps["idx"][idx + 1])
        z = pt.tensor(self.mmaps["z"][atoms], dtype=pt.long)
        pos = pt.tensor(self.mmaps["pos"][atoms])

        props = {}
        if "y" in self.properties:
            props["y"] = pt.tensor(self.mmaps["y"][idx]).view(1, 1)
        if "neg_dy" in self.properties:
            props["neg_dy"] = pt.tensor(self.mmaps["neg_dy"][atoms])
        if "q" in self.properties:
            props["q"] = pt.tensor(self.mmaps["q"][idx], dtype=pt.long)
        if "pq" in self.properties:
            props["pq"] = pt.tensor(self.mmaps["pq"][atoms])
        if "dp" in self.properties:
            props["dp"] = pt.tensor(self.mmaps["dp"][idx])
        return Data(z=z, pos=pos, **props)

    def __del__(self):
        # Flush and close all the memory-mapped files
        import gc

        for prop in list(self.mmaps.keys()):
            self.mmaps[prop].flush()
            del self.mmaps[prop]

        gc.collect()
