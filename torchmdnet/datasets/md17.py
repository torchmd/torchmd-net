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

# extracted from PyG MD17 dataset class


class MD17(InMemoryDataset):

    gdml_url = "http://quantum-machine.org/gdml/data/npz"
    revised_url = (
        "https://archive.materialscloud.org/record/"
        "file?filename=rmd17.tar.bz2&record_id=466"
    )

    file_names = {
        "benzene": "md17_benzene2017.npz",
        "uracil": "md17_uracil.npz",
        "naphtalene": "md17_naphthalene.npz",
        "aspirin": "md17_aspirin.npz",
        "salicylic_acid": "md17_salicylic.npz",
        "malonaldehyde": "md17_malonaldehyde.npz",
        "ethanol": "md17_ethanol.npz",
        "toluene": "md17_toluene.npz",
        "paracetamol": "paracetamol_dft.npz",
        "azobenzene": "azobenzene_dft.npz",
        "revised_benzene": "rmd17_benzene.npz",
        "revised_uracil": "rmd17_uracil.npz",
        "revised_naphthalene": "rmd17_naphthalene.npz",
        "revised_aspirin": "rmd17_aspirin.npz",
        "revised_salicylic_acid": "rmd17_salicylic.npz",
        "revised_malonaldehyde": "rmd17_malonaldehyde.npz",
        "revised_ethanol": "rmd17_ethanol.npz",
        "revised_toluene": "rmd17_toluene.npz",
        "revised_paracetamol": "rmd17_paracetamol.npz",
        "revised_azobenzene": "rmd17_azobenzene.npz",
        "benzene_CCSD_T": "benzene_ccsd_t.zip",
        "aspirin_CCSD": "aspirin_ccsd.zip",
        "malonaldehyde_CCSD_T": "malonaldehyde_ccsd_t.zip",
        "ethanol_CCSD_T": "ethanol_ccsd_t.zip",
        "toluene_CCSD_T": "toluene_ccsd_t.zip",
        "benzene_FHI-aims": "benzene2018_dft.npz",
    }

    def __init__(
        self,
        root: str,
        molecules: str,
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        name = molecules
        if name not in self.file_names:
            raise ValueError(f"Unknown dataset name '{name}'")

        self.name = name
        self.revised = "revised" in name
        self.ccsd = "CCSD" in self.name

        super().__init__(root, transform, pre_transform, pre_filter)

        if len(self.processed_file_names) == 1 and train is not None:
            raise ValueError(
                f"'{self.name}' dataset does not provide pre-defined splits "
                f"but the 'train' argument is set to '{train}'"
            )
        elif len(self.processed_file_names) == 2 and train is None:
            raise ValueError(
                f"'{self.name}' dataset does provide pre-defined splits but "
                f"the 'train' argument was not specified"
            )

        idx = 0 if train is None or train else 1
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def mean(self) -> float:
        return float(self._data.energy.mean())

    @property
    def raw_dir(self) -> str:
        if self.revised:
            return osp.join(self.root, "raw")
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", self.name)

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        if self.revised:
            return osp.join("rmd17", "npz_data", name)
        elif self.ccsd:
            return name[:-4] + "-train.npz", name[:-4] + "-test.npz"
        return name

    @property
    def processed_file_names(self) -> List[str]:
        if self.ccsd:
            return ["train.pt", "test.pt"]
        else:
            return ["data.pt"]

    def download(self):
        if self.revised:
            path = download_url(self.revised_url, self.raw_dir)
            extract_tar(path, self.raw_dir, mode="r:bz2")
            os.unlink(path)
        else:
            url = f"{self.gdml_url}/{self.file_names[self.name]}"
            path = download_url(url, self.raw_dir)
            if self.ccsd:
                extract_zip(path, self.raw_dir)
                os.unlink(path)

    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            if self.revised:
                z = torch.from_numpy(raw_data["nuclear_charges"]).long()
                pos = torch.from_numpy(raw_data["coords"]).float()
                energy = torch.from_numpy(raw_data["energies"]).float()
                force = torch.from_numpy(raw_data["forces"]).float()
            else:
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
