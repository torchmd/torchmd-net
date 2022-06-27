import h5py
import numpy as np
import os
import torch as pt
from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class QM9q(Dataset):

    HARTREE_TO_EV = 27.211386246

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, dataset_arg=None):

        self.name = self.__class__.__name__
        self.dataset_arg = str(dataset_arg)
        super().__init__(root, transform, pre_transform, pre_filter)

        idx_name, z_name, pos_name, y_name, dy_name = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode='r', dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode='r', dtype=np.int8)
        self.pos_mm = np.memmap(pos_name, mode='r', dtype=np.float32, shape=(self.z_mm.shape[0], 3))
        self.y_mm = np.memmap(y_name, mode='r', dtype=np.float64)
        self.dy_mm = np.memmap(dy_name, mode='r', dtype=np.float32, shape=(self.z_mm.shape[0], 3))

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1

    @property
    def raw_paths(self):

        paths = self.dataset_arg

        if os.path.isfile(paths):
            return [paths]
        if os.path.isdir(paths):
           return [os.path.join(paths, file_) for file_ in os.listdir(paths) if file_.endswith('.h5')]

        raise RuntimeError(f'Cannot load {paths}')

    def sample_iter(self):

        for path in tqdm(self.raw_paths, desc='Files'):
            molecules = list(h5py.File(path).values())[0].values()

            for mol in tqdm(molecules, desc='Molecules', leave=False):
                z = pt.tensor(mol['atomic_numbers'], dtype=pt.long)

                for conf in mol['energy']:
                    assert mol['positions'].attrs['units'] == 'Å : ångströms'
                    pos = pt.tensor(mol['positions'][conf], dtype=pt.float32)
                    assert z.shape[0] == pos.shape[0]
                    assert pos.shape[1] == 3

                    assert mol['energy'].attrs['units'] == 'E_h : hartree'
                    y = pt.tensor(mol['energy'][conf][()], dtype=pt.float64) * self.HARTREE_TO_EV

                    #assert mol['gradient_vector'].attrs['units'] == 'vector' # ???
                    dy = -1 * pt.tensor(mol['gradient_vector'][conf], dtype=pt.float32) * self.HARTREE_TO_EV

                    if z.shape[0] != dy.shape[0]: # Skip wrong forces
                        print(f'Skip: mol {mol}, conf {conf}')
                        continue
                    # assert z.shape[0] == dy.shape[0]
                    assert dy.shape[1] == 3

                    data = Data(z=z, pos=pos, y=y.view(1, 1), dy=dy)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    yield data

    @property
    def processed_file_names(self):
        return [f'{self.name}.idx.mmap',
                f'{self.name}.z.mmap',
                f'{self.name}.pos.mmap',
                f'{self.name}.y.mmap',
                f'{self.name}.dy.mmap']

    def process(self):

        print('Gathering statistics...')
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]

        print(f'  Total number of conformers: {num_all_confs}')
        print(f'  Total number of atoms: {num_all_atoms}')

        idx_name, z_name, pos_name, y_name, dy_name = self.processed_paths
        idx_mm = np.memmap(idx_name + '.tmp', mode='w+', dtype=np.int64, shape=(num_all_confs + 1,))
        z_mm = np.memmap(z_name + '.tmp', mode='w+', dtype=np.int8, shape=(num_all_atoms,))
        pos_mm = np.memmap(pos_name + '.tmp', mode='w+', dtype=np.float32, shape=(num_all_atoms, 3))
        y_mm = np.memmap(y_name + '.tmp', mode='w+', dtype=np.float64, shape=(num_all_confs,))
        dy_mm = np.memmap(dy_name + '.tmp', mode='w+', dtype=np.float32, shape=(num_all_atoms,3))

        print('Storing data...')
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y
            dy_mm[i_atom:i_next_atom] = data.dy

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        dy_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(dy_mm.filename, dy_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):

        atoms = slice(self.idx_mm[idx], self.idx_mm[idx+1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(1, 1) # It would be better to use float64, but the trainer complaints
        dy = pt.tensor(self.dy_mm[atoms], dtype=pt.float32)

        return Data(z=z, pos=pos, y=y, dy=dy)