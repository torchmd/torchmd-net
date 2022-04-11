from os.path import join
from urllib import request

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

au_to_eV = 27.211386246


class ANI1X(InMemoryDataset):

    raw_url = "https://figshare.com/ndownloader/files/18112775"

    element_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

    self_energies = {
        "H": -0.500607632585 * au_to_eV,
        "C": -37.8302333826 * au_to_eV,
        "N": -54.5680045287 * au_to_eV,
        "O": -75.0362229210 * au_to_eV,
    }

    @property
    def raw_file_names(self):
        return ["ani1x-release.h5"]

    @property
    def processed_file_names(self):
        return ["ani1x.pt"]

    def __init__(self, root, transform=None, pre_transform=None, **kwargs):
        super(ANI1X, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        raw_archive = join(self.raw_dir, "ani1x-release.h5")
        print(f"Downloading {self.raw_url}")
        request.urlretrieve(self.raw_url, raw_archive)

    @staticmethod
    def iter_data_buckets(h5filename,
                          keys=['wb97x_dz.energy', 'wb97x_dz.forces']):
        """ Iterate over buckets of data in ANI HDF5 file. 
        Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
        and other available properties specified by `keys` list, w/o NaN values.
        Adopted from ANI1x scripts
        """
        keys = set(keys)
        keys.discard('atomic_numbers')
        keys.discard('coordinates')
        with h5py.File(h5filename, 'r') as f:
            for grp in f.values():
                Nc = grp['coordinates'].shape[0]
                mask = np.ones(Nc, dtype=np.bool)
                data = dict((k, grp[k][()]) for k in keys)
                for k in keys:
                    v = data[k].reshape(Nc, -1)
                    mask = mask & ~np.isnan(v).any(axis=1)
                if not np.sum(mask):
                    continue
                d = dict((k, data[k][mask]) for k in keys)
                d['atomic_numbers'] = grp['atomic_numbers'][()]
                d['coordinates'] = grp['coordinates'][()][mask]
                yield d

    def process(self):
        data_list = []
        cnt = 0
        for record in tqdm(self.iter_data_buckets(self.raw_paths[0]),
                           desc="raw h5 files"):
            z = torch.from_numpy(record['atomic_numbers']).long()
            coordinates = torch.from_numpy(record['coordinates']).float()
            energies = torch.from_numpy(
                record['wb97x_dz.energy']).float() * au_to_eV
            forces = torch.from_numpy(
                record['wb97x_dz.forces']).float() * au_to_eV
            for pos, energy, force in zip(coordinates, energies, forces):
                data_list.append(Data(z=z, pos=pos, y=energy.view(1, 1)),
                                 dy=force)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_atomref(self, max_z=100):
        out = torch.zeros(max_z)
        out[list(self.element_numbers.values())] = torch.tensor(
            list(self.self_energies.values()))
        return out.view(-1, 1)
