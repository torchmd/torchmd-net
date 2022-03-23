from glob import glob
import h5py
import torch as pt
from torch_geometric.data import Dataset, Data


class Ace(Dataset):

    def __init__(self, filenames, **kwargs):
        super().__init__()

        # Open HDF5 files
        self.files = [h5py.File(name) for name in glob(filenames)]

        # Index samples
        self.mol_indices = []
        for ifile, file_ in enumerate(self.files):
            for name in file_:
                if 'error' in file_[name]:
                    continue # Skip failed molecules
                num_confs = len(file_[name]['energy'])
                for iconf in range(num_confs):
                    self.mol_indices.append((ifile, name, iconf))

    def get(self, i):

        # Get a molecule
        ifile, name, iconf = self.mol_indices[i]
        mol = self.files[ifile][name]

        # Get molecular data
        return Data(
            q=pt.tensor(mol.attrs['charge'], dtype=pt.long),
            s=pt.tensor(mol.attrs['spin'], dtype=pt.long),
            z=pt.tensor(mol['atomic_numbers'], dtype=pt.long),
            pos=pt.tensor(mol['positions'][iconf], dtype=pt.float32),
            y=pt.tensor(mol['energy'][iconf], dtype=pt.float32),
            dy=pt.tensor(mol['forces'][iconf], dtype=pt.float32),
            d=pt.tensor(mol['dipole_moment'][iconf], dtype=pt.float32))

    def len(self):
        return len(self.mol_indices)

    def get_atomref(self):
        raise NotImplementedError