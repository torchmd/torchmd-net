import h5py
import torch as pt
from torch_geometric.data import Dataset, Data


class Ace(Dataset):

    def __init__(self, filenames):
        super().__init__()

        # Open HDF5 files
        self.files = [h5py.File(name) for name in filenames]

        # Index samples
        self.mol_indices = []
        for ifile, file_ in enumerate(self.files):
            for mol_name in file_:
                if 'error' in file_[mol_name]:
                    continue # Skip failed molecules
                num_confs = len(file_[mol_name]['energy'])
                for iconf in range(num_confs):
                    self.mol_indices.append((ifile, mol_name, iconf))

    def get(self, i):

        # Get a molecule
        ifile, mol_name, iconf = self.mol_indices[i]
        mol = self.files[ifile][mol_name]

        # Get molecular data
        return Data(
            q=pt.from_numpy(mol.attrs['charge']),
            s=pt.from_numpy(mol.attrs['spin']),
            z=pt.from_numpy(mol['atomic_numbers']),
            pos=pt.from_numpy(mol['positions'][iconf]),
            y=pt.from_numpy(mol['energy'][iconf]),
            dy=pt.from_numpy(mol['forces'][iconf]),
            d=pt.from_numpy(mol['dipole_moment'][iconf]))

    def len(self):
        return len(self.mol_indices)