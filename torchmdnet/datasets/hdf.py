import torch
from torch_geometric.data import Dataset, Data
import h5py
import numpy as np


class HDF5(Dataset):
    """A custom dataset that loads data from a HDF5 file.

    To use this, dataset_root should be the path to the HDF5 file, or alternatively
    a semicolon separated list of multiple files.  Each group in the file contains
    samples that all have the same number of atoms.  Typically there is one
    group for each unique number of atoms, but that is not required.  Each group
    should contain arrays called "types" (atom type indices), "pos" (atom positions),
    and "energy" (the energy of each sample).  It may optionally include an array
    called "forces" (the force on each atom).

    Args:
        filename (string): A semicolon separated list of HDF5 files.
    """

    def __init__(self, filename, **kwargs):
        super(HDF5, self).__init__()
        self.filename = filename
        self.index = None
        self.fields = None
        self.num_molecules = 0
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        for file in files:
            for group_name in file:
                group = file[group_name]
                if group_name == '_metadata':
                    for name in group:
                        setattr(self, name, torch.tensor(np.array(group[name])))
                else:
                    self.num_molecules += len(group["energy"])
                    if self.fields is None:
                        # Record which data fields are present in this file.
                        self.fields = [
                            ('pos', 'pos', torch.float32),
                            ('z', 'types', torch.long),
                            ('y', 'energy', torch.float32)
                        ]
                        if 'forces' in group:
                            self.fields.append(('neg_dy', 'forces', torch.float32))
                        if 'partial_charges' in group:
                            self.fields.append(('partial_charges', 'partial_charges', torch.float32))
            file.close()

    def setup_index(self):
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        self.has_forces = False
        self.index = []
        for file in files:
            for group_name in file:
                if group_name != '_metadata':
                    group = file[group_name]
                    data = tuple(group[field[1]] for field in self.fields)
                    energy = group['energy']
                    for i in range(len(energy)):
                        self.index.append(data+(i,))

        assert self.num_molecules == len(self.index), (
            "Mismatch between previously calculated "
            "molecule count and actual molecule count"
        )

    def get(self, idx):
        # only open files here to avoid copying objects of this class to another
        # process with open file handles (potentially corrupts h5py loading)
        if self.index is None:
            self.setup_index()

        entry = self.index[idx]
        i = entry[-1]
        data = Data()
        for j, field in enumerate(self.fields):
            d = entry[j]
            if d.ndim == 1:
                data[field[0]] = torch.tensor([[d[i]]], dtype=field[2])
            else:
                data[field[0]] = torch.from_numpy(d[i]).to(field[2])
        return data

    def len(self):
        return self.num_molecules
