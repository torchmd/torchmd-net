import h5py
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

        self.h5 = h5py.File(self.processed_paths[0])
        self.ds_z = self.h5['z']
        self.ds_pos = self.h5['pos']
        self.ds_y = self.h5['y']

    @property
    def raw_file_names(self):
        return [os.path.join('ANI-1_release', f'ani_gdb_s{i:02d}.h5') for i in range(1, 9)]

    @property
    def processed_file_names(self):
        return ['ani-1.h5']

    def download(self):
        archive = download_url(self.raw_url, self.raw_dir)
        extract_tar(archive, self.raw_dir)
        os.remove(archive)

    def process(self):

        tmp_file = self.processed_paths[0] + '.tmp'
        with h5py.File(tmp_file, 'w', driver='core') as h5:

            ds_z = h5.create_dataset('z', (10000,), chunks=(10000,), maxshape=(None,),
                                     dtype=h5py.vlen_dtype('int8'))
            ds_pos = h5.create_dataset('pos', (10000,), chunks=(10000,), maxshape=(None,),
                                       dtype=h5py.vlen_dtype('float32'))
            ds_y = h5.create_dataset('y', (10000,), chunks=(10000,), maxshape=(None,),
                                     dtype='float64')

            i_sample = 0

            for path in tqdm(self.raw_paths[7:], desc='Files'):
                molecules = list(h5py.File(path).values())[0].values()

                for mol in tqdm(molecules, desc='Molecules', leave=False):
                    z = pt.tensor([self.atomic_numbers[atom] for atom in mol['species']])
                    all_pos = pt.tensor(mol['coordinates'][:])
                    all_y = pt.tensor(mol['energies'][:] * self.HARTREE_TO_EV)

                    assert all_pos.shape[0] == all_y.shape[0]
                    assert all_pos.shape[1] == z.shape[0]
                    assert all_pos.shape[2] == 3

                    for pos, y in zip(all_pos, all_y):
                        data = Data(z=z, pos=pos, y=y.view(1, 1))

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        if i_sample >= len(ds_z):
                            ds_z.resize(i_sample + 10000, axis=0)
                            ds_pos.resize(i_sample + 10000, axis=0)
                            ds_y.resize(i_sample + 10000, axis=0)

                        ds_z[i_sample] = data.z
                        ds_pos[i_sample] = data.pos.flatten()
                        ds_y[i_sample] = data.y

                        i_sample += 1

            ds_z.resize(i_sample, axis=0)
            ds_pos.resize(i_sample, axis=0)
            ds_y.resize(i_sample, axis=0)

        os.rename(tmp_file, self.processed_paths[0])

    def len(self):
        return len(self.ds_z)

    def get(self, idx):

        z = pt.tensor(self.ds_y[idx])
        pos = pt.tensor(self.ds_pos[idx]).view(-1, 3)
        y = pt.tensor(self.ds_y[idx]).view(1, 1)

        return Data(z=z, pos=pos, y=y)

    def get_atomref(self, max_z=100):

        out = pt.zeros(max_z)
        out[list(self.atomic_numbers.values())] = pt.tensor(list(self.atomic_energies.values()))

        return out.view(-1, 1)