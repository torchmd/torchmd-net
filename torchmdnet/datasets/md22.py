# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)


class MD22(InMemoryDataset):

    gdml_url = "http://quantum-machine.org/gdml/data/npz"

    file_names = {
        "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
        "AT-AT": "md22_AT-AT.npz",
        "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
        "DHA": "md22_DHA.npz",
        "buckyball-catcher": "md22_buckyball-catcher.npz",
        "dw-nanotube": "md22_dw_nanotube.npz",
        "stachyose": "md22_stachyose.npz",
    }

    def __init__(
        self,
        root: str,
        molecules: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        name = molecules
        if name not in self.file_names:
            raise ValueError(f"Unknown dataset name '{name}'")

        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)

        idx = 0
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def mean(self) -> float:
        return float(self._data.energy.mean())

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", self.name)

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        url = f"{self.gdml_url}/{self.file_names[self.name]}"
        path = download_url(url, self.raw_dir)

    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            z = torch.from_numpy(raw_data["z"]).long()
            pos = torch.from_numpy(raw_data["R"]).float()
            energy = torch.from_numpy(raw_data["E"]).float()
            force = torch.from_numpy(raw_data["F"]).float()

            data_list = []
            for i in range(pos.size(0)):
                data = Data(z=z, pos=pos[i], y=energy[i].unsqueeze(-1), neg_dy=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
