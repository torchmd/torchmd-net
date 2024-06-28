# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import hashlib
import h5py
import numpy as np
import os
import torch as pt
from torchmdnet.datasets.memdataset import MemmappedDataset
from torch_geometric.data import Data, download_url
from tqdm import tqdm
import logging


class SPICE(MemmappedDataset):
    """
    SPICE dataset (https://github.com/openmm/spice-dataset)

    The dataset has several versions (https://github.com/openmm/spice-dataset/releases).
    The version can be selected with `version`. By default, version 1.1.3 is loaded.

    >>> ds = SPICE(".", version="1.1.3")

    The dataset consists of several subsets (https://github.com/openmm/spice-dataset/blob/main/downloader/config.yaml).
    The subsets can be selected with `subsets`. By default, all the subsets are loaded.

    For example, this loads just two subsets:
    >>> ds = SPICE(".", subsets=["SPICE PubChem Set 1 Single Points Dataset v1.2", "SPICE PubChem Set 2 Single Points Dataset v1.2"])

    The loader can filter conformations with large gradients. The maximum gradient norm threshold
    can be set with `max_gradient`. By default, the filter is not applied.

    For example, the filter the threshold is set to 100 eV/A:
    >>> ds = SPICE(".", max_gradient=100)

    The molecules can be subsampled by loading only every `subsample_molecules`-th molecule.
    By default is `subsample_molecules` is set to 1 (load all the molecules).

    For example, only every 10th molecule is loaded:
    >>> ds = SPICE(".", subsample_molecules=10)
    """

    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177

    VERSIONS = {
        "1.0": {
            "url": "https://github.com/openmm/spice-dataset/releases/download/1.0",
            "file": "SPICE.hdf5",
        },
        "1.1": {
            "url": "https://github.com/openmm/spice-dataset/releases/download/1.1",
            "file": "SPICE.hdf5",
        },
        "1.1.1": {
            "url": "https://zenodo.org/record/7258940/files",
            "file": "SPICE-1.1.1.hdf5",
            "hash": "5411e7014c6d18ff07d108c9ad820b53",
        },
        "1.1.2": {
            "url": "https://zenodo.org/record/7338495/files",
            "file": "SPICE-1.1.2.hdf5",
            "hash": "a2b5ae2d1f72581040e1cceb20a79a33",
        },
        "1.1.3": {
            "url": "https://zenodo.org/record/7606550/files",
            "file": "SPICE-1.1.3.hdf5",
            "hash": "be93706b3bb2b2e327b690b185905856",
        },
        "1.1.4": {
            "url": "https://zenodo.org/records/8222043/files",
            "file": "SPICE-1.1.4.hdf5",
            "hash": "f27d4c81da0e37d6547276bf6b4ae6a1",
        },
        "2.0.1": {
            "url": "https://zenodo.org/records/10975225/files",
            "file": "SPICE-2.0.1.hdf5",
            "hash": "bfba2224b6540e1390a579569b475510",
        },
    }

    @property
    def raw_dir(self):
        return os.path.join(super().raw_dir, "spice", self.version)

    @property
    def raw_file_names(self):
        return self.VERSIONS[self.version]["file"]

    @property
    def raw_url(self):
        return f"{self.VERSIONS[self.version]['url']}/{self.VERSIONS[self.version]['file']}"

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="1.1.3",
        subsets=None,
        max_gradient=None,
        subsample_molecules=1,
    ):
        arg_hash = f"{version}{subsets}{max_gradient}{subsample_molecules}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.version = str(version)
        assert self.version in self.VERSIONS
        self.subsets = subsets
        self.max_gradient = max_gradient
        self.subsample_molecules = int(subsample_molecules)
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y", "neg_dy"),
        )

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1
        assert self.subsample_molecules > 0

        molecules = h5py.File(self.raw_paths[0]).items()
        for i_mol, (mol_id, mol) in tqdm(enumerate(molecules), desc="Molecules"):
            if self.subsets:
                if mol["subset"][0].decode() not in list(self.subsets):
                    continue

            # Subsample molecules
            if i_mol % self.subsample_molecules != 0:
                continue

            z = pt.tensor(np.array(mol["atomic_numbers"]), dtype=pt.long)
            all_pos = (
                pt.tensor(np.array(mol["conformations"]), dtype=pt.float32)
                * self.BORH_TO_ANGSTROM
            )
            all_y = (
                pt.tensor(np.array(mol["formation_energy"]), dtype=pt.float64)
                * self.HARTREE_TO_EV
            )
            all_neg_dy = (
                -pt.tensor(np.array(mol["dft_total_gradient"]), dtype=pt.float32)
                * self.HARTREE_TO_EV
                / self.BORH_TO_ANGSTROM
            )
            if all_pos.ndim < 3:
                logging.warning(f"Bogus conformation {mol_id}")
                logging.warning(
                    f"Found {all_pos.shape} positions, {all_y.shape} energies and {all_neg_dy.shape} gradients"
                )
                continue
            assert all_pos.shape[0] == all_y.shape[0]
            assert all_pos.shape[1] == z.shape[0]
            assert all_pos.shape[2] == 3

            assert all_neg_dy.shape[0] == all_y.shape[0]
            assert all_neg_dy.shape[1] == z.shape[0]
            assert all_neg_dy.shape[2] == 3

            for pos, y, neg_dy in zip(all_pos, all_y, all_neg_dy):
                # Skip samples with large forces
                if self.max_gradient:
                    if neg_dy.norm(dim=1).max() > float(self.max_gradient):
                        continue

                # Create a sample
                args = dict(z=z, pos=pos, y=y.view(1, 1), neg_dy=neg_dy)
                if mol_ids:
                    args["mol_id"] = mol_id
                data = Data(**args)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                yield data

    def download(self):
        download_url(self.raw_url, self.raw_dir)
        if "hash" in self.VERSIONS[self.version]:
            with open(self.raw_paths[0], "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            assert file_hash == self.VERSIONS[self.version]["hash"]
