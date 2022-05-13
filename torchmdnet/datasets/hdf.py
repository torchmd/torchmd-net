import torch
from torch_geometric.data import Dataset, Data
import h5py


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

        self.num_molecules = 0
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        for file in files:
            for group_name in file:
                self.num_molecules += len(file[group_name]["energy"])
            file.close()

    def setup_index(self):
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        self.has_forces = False
        self.index = []
        for file in files:
            for group_name in file:
                group = file[group_name]
                types = group["types"]
                pos = group["pos"]
                energy = group["energy"]
                if "forces" in group:
                    self.has_forces = True
                    forces = group["forces"]
                    for i in range(len(energy)):
                        self.index.append((types, pos, energy, forces, i))
                else:
                    for i in range(len(energy)):
                        self.index.append((types, pos, energy, i))

        assert self.num_molecules == len(self.index), (
            "Mismatch between previously calculated "
            "molecule count and actual molecule count"
        )

    def get(self, idx):
        # only open files here to avoid copying objects of this class to another
        # process with open file handles (potentially corrupts h5py loading)
        if self.index is None:
            self.setup_index()

        if self.has_forces:
            types, pos, energy, forces, i = self.index[idx]
            return Data(
                pos=torch.from_numpy(pos[i]),
                z=torch.from_numpy(types[i]).to(torch.long),
                y=torch.tensor([[energy[i]]]),
                dy=torch.from_numpy(forces[i]),
            )
        else:
            types, pos, energy, i = self.index[idx]
            return Data(
                pos=torch.from_numpy(pos[i]),
                z=torch.from_numpy(types[i]).to(torch.long),
                y=torch.tensor([[energy[i]]]),
            )

    def len(self):
        return self.num_molecules
