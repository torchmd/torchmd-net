import hashlib
import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class Ace(Dataset):

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        paths=None,
        max_gradient=None,
    ):

        arg_hash = f"{paths}{max_gradient}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.paths = str(paths)
        self.max_gradients = float(max_gradient)
        super().__init__(root, transform, pre_transform, pre_filter)

        (
            idx_name,
            z_name,
            pos_name,
            y_name,
            dy_name,
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
        self.dy_mm = np.memmap(
            dy_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
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

        if os.path.isfile(self.paths):
            return [self.paths]
        if os.path.isdir(self.paths):
            return [
                os.path.join(self.paths, file_)
                for file_ in os.listdir(self.paths)
                if file_.endswith(".h5")
            ]

        raise RuntimeError(f"Cannot load {self.paths}")

    def sample_iter(self):

        for path in tqdm(self.raw_paths, desc="Files"):
            molecules = list(h5py.File(path).values())

            for mol in tqdm(molecules, desc="Molecules", leave=False):
                z = pt.tensor(mol["atomic_numbers"], dtype=pt.long)
                fq = pt.tensor(mol["formal_charges"], dtype=pt.long)
                q = fq.sum()

                for conf in mol["conformations"].values():

                    assert conf["positions"].attrs["units"] == "Å"
                    pos = pt.tensor(conf["positions"], dtype=pt.float32)
                    assert pos.shape == (z.shape[0], 3)

                    assert conf["formation_energy"].attrs["units"] == "eV"
                    y = pt.tensor(conf["formation_energy"][()], dtype=pt.float64)
                    assert y.shape == (1,)

                    assert conf["forces"].attrs["units"] == "eV/Å"
                    dy = -pt.tensor(conf["forces"][conf], dtype=pt.float32)
                    assert dy.shape == pos.shape

                    assert conf["partial_charges"].attrs["units"] == "e"
                    pq = pt.tensor(conf["partial_charges"], dtype=pt.float32)
                    assert pq.shape == z.shape

                    assert conf["dipole_moment"].attrs["units"] == "e*Å"
                    dp = pt.tensor(conf["dipole_moment"], dtype=pt.float32)
                    assert dp.shape == (3,)

                    # Skip samples with large forces
                    if self.max_gradient:
                        if dy.norm(dim=1).max() > float(self.max_gradient):
                            continue

                    data = Data(z=z, pos=pos, y=y.view(1, 1), dy=dy, q=q, pq=pq, dp=dp)

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
            f"{self.name}.dy.mmap",
            f"{self.name}.q.mmap",
            f"{self.name}.pq.mmap",
            f"{self.name}.dp.mmap",
        ]

    def process(self):

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
            dy_name,
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
        dy_mm = np.memmap(
            dy_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
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
            dy_mm[i_atom:i_next_atom] = data.dy
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
        dy_mm.flush()
        q_mm.flush()
        pq_mm.flush()
        dp_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(dy_mm.filename, dy_name)
        os.rename(q_mm.filename, q_name)
        os.rename(pq_mm.filename, pq_name)
        os.rename(dp_mm.filename, dp_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):

        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        dy = pt.tensor(self.dy_mm[atoms], dtype=pt.float32)
        q = pt.tensor(self.q_mm[idx], dtype=pt.long)
        pq = pt.tensor(self.pq_mm[atoms], dtype=pt.float32)
        dp = pt.tensor(self.dp_mm[idx], dtype=pt.float32)

        return Data(z=z, pos=pos, y=y, dy=dy, q=q, pq=pq, dp=dp)
