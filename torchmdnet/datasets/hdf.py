# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch_geometric.data import Dataset, Data
import h5py
import numpy as np


class HDF5(Dataset):
    """A custom dataset that loads data from a HDF5 file.

    To use this, dataset_root should be  the path to the HDF5 file, or
    alternatively a semicolon separated  list of multiple files.  Each
    group in the  file contains samples that all have  the same number
    of atoms.  Typically there is one  group for each unique number of
    atoms, but that is not required.  Each group should contain arrays
    called "types" (atom type indices), "pos" (atom positions), and at
    least one of "energy" (the  energy of each sample) and/or "forces"
    (the force on each atom).

    Args:
        filename (string): A semicolon separated list of HDF5 files.
        dataset_preload_limit (int, optional): If the dataset is smaller than this limit (in MB), preload it into CPU memory. (default: :obj:`1024`)

    """

    def __init__(self, filename, dataset_preload_limit=1024, **kwargs):
        super(HDF5, self).__init__()
        self.filename = filename
        self.index = None
        self.fields = None
        self.num_molecules = 0
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        total_file_size = sum([f.id.get_filesize() for f in files])
        print(f"Loading {len(files)} HDF5 files ({total_file_size / 1024**2:.2f} MB)")
        for file in files:
            for group_name in file:
                group = file[group_name]
                if group_name == "_metadata":
                    for name in group:
                        setattr(self, name, torch.tensor(np.array(group[name])))
                else:
                    self.num_molecules += len(group["pos"])
                    if self.fields is None:
                        # Record which data fields are present in this file.
                        self.fields = [
                            ("pos", "pos", torch.float32),
                            ("z", "types", torch.long),
                        ]
                        if "energy" in group:
                            self.fields.append(("y", "energy", torch.float32))
                        if "forces" in group:
                            self.fields.append(("neg_dy", "forces", torch.float32))
                        if "partial_charges" in group:
                            self.fields.append(
                                ("partial_charges", "partial_charges", torch.float32)
                            )
                        assert ("energy" in group) or (
                            "forces" in group
                        ), "Each group must contain at least energies or forces"
            file.close()
        self.cached = False
        if total_file_size <= dataset_preload_limit * 1024**2:
            print(
                f"Preloading {len(files)} HDF5 files ({total_file_size / 1024**2:.2f} MB)"
            )
            self.cached = True
            self._preload_data()

    def _preload_data(self):
        """Preload the entire dataset into memory.
        Store it in a dictionary of torch tensors. The dictionary has an entry for each field and group.
        """
        self.stored_data = {}
        for field in self.fields:
            self.stored_data[field] = []
        self.index = []
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        i = 0
        for file in files:
            for group_name, group in file.items():
                if group_name != "_metadata":
                    size = len(group["pos"])
                    for field in self.fields:
                        data = group[field[1]]
                        dtype = field[2]
                        # Watchout for the 1D case, embed can be shared for all samples
                        tmp = torch.tensor(np.array(data), dtype=dtype)
                        if tmp.ndim == 1:
                            if len(tmp) == size:
                                tmp = tmp.unsqueeze(-1)
                            else:
                                tmp = tmp.unsqueeze(0).expand(size, -1)
                        self.stored_data[field].append(tmp)
                    self.index.extend(list(zip([i] * size, range(size))))
                    i += 1
            file.close()

    def _setup_index(self):
        files = [h5py.File(f, "r") for f in self.filename.split(";")]
        self.index = []
        for file in files:
            for group_name, group in file.items():
                if group_name != "_metadata":
                    data = [group[field[1]] for field in self.fields]
                    self.index.extend(
                        [tuple(data + [i]) for i in range(len(group["pos"]))]
                    )
        assert self.num_molecules == len(
            self.index
        ), f"Mismatch between previously calculated molecule count ({self.num_molecules}) and actual molecule count ({len(self.index)})"

    def get(self, idx):
        data = Data()
        if self.cached:
            # If the dataset is cached, the index is just a list of indices into the stored data.
            fileid, index = self.index[idx]
            for field in self.fields:
                data[field[0]] = self.stored_data[field][fileid][index]
        else:
            # only open files here to avoid copying objects of this class to another
            # process with open file handles (potentially corrupts h5py loading)
            if self.index is None:
                self._setup_index()
            *fields_data, i = self.index[idx]
            # Assuming the first element of fields_data is 'pos' based on the definition of self.fields
            size = len(fields_data[0])
            for (name, _, dtype), d in zip(self.fields, fields_data):
                if d.ndim == 1:
                    tensor_input = [d[i]] if len(d) == size else d[:]
                else:
                    tensor_input = d[i]
                data[name] = torch.tensor(tensor_input, dtype=dtype)
        return data

    def len(self):
        return self.num_molecules
