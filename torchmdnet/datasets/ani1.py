import os
from os.path import join
from tqdm import tqdm
from urllib import request
import torch
from torch_geometric.data import InMemoryDataset, extract_tar, Data
import h5py


class ANI1(InMemoryDataset):

    raw_url = "https://ndownloader.figshare.com/files/9057631"

    element_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

    HAR2EV = 27.211386246

    self_energies = {
        "H": -0.500607632585 * HAR2EV,
        "C": -37.8302333826 * HAR2EV,
        "N": -54.5680045287 * HAR2EV,
        "O": -75.0362229210 * HAR2EV,
    }

    def __init__(self, root, transform=None, pre_transform=None, **kwargs):
        super(ANI1, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"ANI-1_release/ani_gdb_s{i + 1:02d}.h5" for i in range(8)]

    @property
    def processed_file_names(self):
        return ["ani1.pt"]

    def download(self):
        raw_archive = join(self.raw_dir, "ANI1_release.tar.gz")
        print(f"Downloading {self.raw_url}")
        request.urlretrieve(self.raw_url, raw_archive)
        extract_tar(raw_archive, self.raw_dir)
        os.remove(raw_archive)

    def process(self):
        data_list = []
        for path in tqdm(self.raw_paths, desc="raw h5 files"):
            data = h5py.File(path, "r")
            for file_name in data:
                for molecule_name in tqdm(
                    data[file_name], desc="molecules", leave=False
                ):
                    group = data[file_name][molecule_name]
                    elements = torch.tensor(
                        [
                            self.element_numbers[str(elem)[-2]]
                            for elem in group["species"]
                        ]
                    )
                    positions = torch.from_numpy(group["coordinates"][:])
                    energies = torch.from_numpy(
                        group["energies"][:] * self.HAR2EV
                    ).float()

                    elements = elements.expand(positions.size(0), -1)
                    for z, pos, energy in zip(elements, positions, energies):
                        data_list.append(Data(z=z, pos=pos, y=energy.view(1, 1)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_atomref(self, max_z=100):
        out = torch.zeros(max_z)
        out[list(self.element_numbers.values())] = torch.tensor(
            list(self.self_energies.values())
        )
        return out.view(-1, 1)
