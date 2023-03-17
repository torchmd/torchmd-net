import hashlib
import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, Dataset, download_url
from tqdm import tqdm


class SPICE(Dataset):

    """
    SPICE dataset (https://github.com/openmm/spice-dataset)

    The dataset has several versions (https://github.com/openmm/spice-dataset/releases).
    The version can be selected with `version`. By default, version 1.1.3 is loaded.

    >>> ds = SPICE(".", version="1.1.3")

    The dataset consists of several subsets (https://github.com/openmm/spice-dataset/blob/main/downloader/config.yaml).
    The subsets can be selected with `subsets`. By default, all the subsets are loaded.

    For example, this loads just two subsets:
    >>> ds = SPICE(".", subsets=["SPICE PubChem Set 1 Single Points Dataset v1.2", "SPICE PubChem Set 2 Single Points Dataset v1.2"])

    The loader can filter conformations with large gradients. The maximum gradient norm threshold
    can be set with `max_gradient`. By default, the filter is not applied.

    For example, the filter the threshold is set to 100 eV/A:
    >>> ds = SPICE(".", max_gradient=100)

    The molecules can be subsampled by loading only every `subsample_molecules`-th molecule.
    By default is `subsample_molecules` is set to 1 (load all the molecules).

    For example, only every 10th molecule is loaded:
    >>> ds = SPICE(".", subsample_molecules=10)
    """

    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177

    VERSIONS = {
        "1.0": {"url": "https://github.com/openmm/spice-dataset/releases/download/1.0", "file": "SPICE.hdf5"},
        "1.1": {"url": "https://github.com/openmm/spice-dataset/releases/download/1.1", "file": "SPICE.hdf5"},
        "1.1.1": {"url": "https://zenodo.org/record/7258940/files", "file": "SPICE-1.1.1.hdf5"},
        "1.1.2": {"url": "https://zenodo.org/record/7338495/files", "file": "SPICE-1.1.2.hdf5"},
        "1.1.3": {"url": "https://zenodo.org/record/7606550/files", "file": "SPICE-1.1.3.hdf5"},
    }

    @property
    def raw_dir(self):
        return os.path.join(super().raw_dir, self.version)

    @property
    def raw_file_names(self):
        return self.VERSIONS[self.version]["file"]

    @property
    def raw_url(self):
        return f"{self.VERSIONS[self.version]['url']}/{self.VERSIONS[self.version]['file']}"

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx.mmap",
            f"{self.name}.z.mmap",
            f"{self.name}.pos.mmap",
            f"{self.name}.y.mmap",
            f"{self.name}.neg_dy.mmap",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="1.1.3",
        subsets=None,
        max_gradient=None,
        subsample_molecules=1,
    ):
        arg_hash = f"{version}{subsets}{max_gradient}{subsample_molecules}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.version = str(version)
        assert self.version in self.VERSIONS
        self.subsets = subsets
        self.max_gradient = max_gradient
        self.subsample_molecules = int(subsample_molecules)
        super().__init__(root, transform, pre_transform, pre_filter)

        idx_name, z_name, pos_name, y_name, neg_dy_name = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)
        self.neg_dy_mm = np.memmap(
            neg_dy_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1
        assert self.subsample_molecules > 0

        molecules = h5py.File(self.raw_paths[0]).items()
        for i_mol, (mol_id, mol) in tqdm(enumerate(molecules), desc="Molecules"):

            if self.subsets:
                if mol["subset"][0].decode() not in list(self.subsets):
                    continue

            # Subsample molecules
            if i_mol % self.subsample_molecules != 0:
                continue

            z = pt.tensor(mol["atomic_numbers"], dtype=pt.long)
            all_pos = (
                pt.tensor(mol["conformations"], dtype=pt.float32)
                * self.BORH_TO_ANGSTROM
            )
            all_y = (
                pt.tensor(mol["formation_energy"], dtype=pt.float64)
                * self.HARTREE_TO_EV
            )
            all_neg_dy = (
                -pt.tensor(mol["dft_total_gradient"], dtype=pt.float32)
                * self.HARTREE_TO_EV
                / self.BORH_TO_ANGSTROM
            )

            assert all_pos.shape[0] == all_y.shape[0]
            assert all_pos.shape[1] == z.shape[0]
            assert all_pos.shape[2] == 3

            assert all_neg_dy.shape[0] == all_y.shape[0]
            assert all_neg_dy.shape[1] == z.shape[0]
            assert all_neg_dy.shape[2] == 3

            for pos, y, neg_dy in zip(all_pos, all_y, all_neg_dy):

                # Skip samples with large forces
                if self.max_gradient:
                    if neg_dy.norm(dim=1).max() > float(self.max_gradient):
                        continue

                # Create a sample
                args = dict(z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy)
                if mol_ids:
                    args["mol_id"] = mol_id
                data = Data(**args)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                yield data

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    def process(self):

        print("Arguments")
        print(f"  version: {self.version}")
        print(f"  subsets: {self.subsets}")
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
        neg_dy_mm = np.memmap(
            neg_dy_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
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

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        neg_dy_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(neg_dy_mm.filename, neg_dy_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        neg_dy = pt.tensor(self.neg_dy_mm[atoms], dtype=pt.float32)

        return Data(z=z, pos=pos, y=y, neg_dy=neg_dy)
