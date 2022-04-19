import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, Dataset, download_url, extract_tar
from tqdm import tqdm


class ANI1(Dataset):

    HARTREE_TO_EV = 27.211386246

    raw_url = 'https://ndownloader.figshare.com/files/9057631'
    atomic_numbers = {b'H': 1, b'C': 6, b'N': 7, b'O': 8}
    atomic_energies = {
        'H': -0.500607632585 * HARTREE_TO_EV,
        'C': -37.8302333826 * HARTREE_TO_EV,
        'N': -54.5680045287 * HARTREE_TO_EV,
        'O': -75.0362229210 * HARTREE_TO_EV,
    }

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        idx_name, z_name, pos_name, y_name = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode='r', dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode='r', dtype=np.int8)
        self.pos_mm = np.memmap(pos_name, mode='r', dtype=np.float32, shape=(self.z_mm.shape[0], 3))
        self.y_mm = np.memmap(y_name, mode='r', dtype=np.float64)

        print(self.z_mm.shape)
        print(self.pos_mm.shape)

    @property
    def raw_file_names(self):
        return [os.path.join('ANI-1_release', f'ani_gdb_s{i:02d}.h5') for i in range(1, 9)]

    @property
    def processed_file_names(self):
        return ['ani-1.idx.mmap', 'ani-1.z.mmap', 'ani-1.pos.mmap', 'ani-1.y.mmap']

    def download(self):
        archive = download_url(self.raw_url, self.raw_dir)
        extract_tar(archive, self.raw_dir)
        os.remove(archive)

    def _sample_iter(self):

        for path in tqdm(self.raw_paths, desc='Files'):
            molecules = list(h5py.File(path).values())[0].values()

            for mol in tqdm(molecules, desc='Molecules', leave=False):
                z = pt.tensor([self.atomic_numbers[atom] for atom in mol['species']], dtype=pt.long)
                all_pos = pt.tensor(mol['coordinates'][:], dtype=pt.float32)
                all_y = pt.tensor(mol['energies'][:] * self.HARTREE_TO_EV, dtype=pt.float64)

                assert all_pos.shape[0] == all_y.shape[0]
                assert all_pos.shape[1] == z.shape[0]
                assert all_pos.shape[2] == 3

                for pos, y in zip(all_pos, all_y):
                    data = Data(z=z, pos=pos, y=y.view(1, 1))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    yield data

    def process(self):

        print('Gather statistics...')
        num_all_confs = 0
        num_all_atoms = 0
        for data in self._sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]

        print(f'  Total number of conformers: {num_all_confs}')
        print(f'  Total number of atoms: {num_all_atoms}')

        idx_name, z_name, pos_name, y_name = self.processed_paths
        idx_mm = np.memmap(idx_name + '.tmp', mode='w+', dtype=np.int64, shape=(num_all_confs + 1,))
        z_mm = np.memmap(z_name + '.tmp', mode='w+', dtype=np.int8, shape=(num_all_atoms,))
        pos_mm = np.memmap(pos_name + '.tmp', mode='w+', dtype=np.float32, shape=(num_all_atoms, 3))
        y_mm = np.memmap(y_name + '.tmp', mode='w+', dtype=np.float64, shape=(num_all_confs,))

        print('Storing data...')
        i_atom = 0
        for i_conf, data in enumerate(self._sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):

        i_fist_atom, i_last_atom = self.idx_mm[idx], self.idx_mm[idx+1]
        z = pt.tensor(self.z_mm[i_fist_atom:i_last_atom], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[i_fist_atom:i_last_atom], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float64).view(1, 1)

        return Data(z=z, pos=pos, y=y)

    def get_atomref(self, max_z=100):

        out = pt.zeros(max_z)
        out[list(self.atomic_numbers.values())] = pt.tensor(list(self.atomic_energies.values()))

        return out.view(-1, 1)