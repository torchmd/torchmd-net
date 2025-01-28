# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import hashlib
from torchmdnet.utils import ATOMIC_NUMBERS
import numpy as np
import os
import torch as pt
from torchmdnet.datasets.memdataset import MemmappedDataset
from torch_geometric.data import Data, download_url
import tarfile
import logging
import re
from tqdm import tqdm


def parse_maceoff_tar(tar_file):
    energy_re = re.compile("energy=(\S+)")
    with tarfile.open(tar_file, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            n_atoms = None
            counter = 0
            positions = []
            numbers = []
            forces = []
            energy = None
            for line in f:
                line = line.decode("utf-8").strip()
                if n_atoms is None:
                    n_atoms = int(line)
                    positions = []
                    numbers = []
                    forces = []
                    energy = None
                    counter = 1
                    continue
                if counter == 1:
                    props = line
                    energy = float(energy_re.search(props).group(1))
                    counter = 2
                    continue
                el, x, y, z, fx, fy, fz, _, _, _ = line.split()
                numbers.append(ATOMIC_NUMBERS[el])
                positions.append([float(x), float(y), float(z)])
                forces.append([float(fx), float(fy), float(fz)])
                counter += 1
                if counter == n_atoms + 2:
                    n_atoms = None
                    yield energy, numbers, positions, forces


class MACEOFF(MemmappedDataset):
    """
        MACEOFF dataset from MACE-OFF23: Transferable Machine Learning Force Fields for Organic Molecules, Kovacs et.al. https://arxiv.org/abs/2312.15211
        This dataset consists of arounf 100K conformations with 95% of them coming from SPICE and augmented with conformations from QMugs, COMP6 and clusters of water carved out of MD simulations of liquid water.

        From the repository:
        The core of the training set is the SPICE dataset. 95% of the data were used for training and validation, and 5% for testing. The MACE-OFF23 model is trained to reproduce the energies and forces computed at the ωB97M-D3(BJ)/def2-TZVPPD level of quantum mechanics, as implemented in the PSI4 software. We have used a subset of SPICE that contains the ten chemical elements H, C, N, O, F, P, S, Cl, Br, and I, and has a neutral formal charge. We have also removed the ion pairs subset. Overall, we used about 85% of the full SPICE dataset.

    Contains energy and force data in units of eV and eV/Angstrom

    """

    VERSIONS = {
        "1.0": {
            "url": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/b185b5ab-91cf-489a-9302-63bfac42824a/content",
            "file": "train_large_neut_no_bad_clean.tar.gz",
        },
    }

    @property
    def raw_dir(self):
        return os.path.join(super().raw_dir, "maceoff", self.version)

    @property
    def raw_file_names(self):
        return self.VERSIONS[self.version]["file"]

    @property
    def raw_url(self):
        return f"{self.VERSIONS[self.version]['url']}"

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        version="1.0",
        max_gradient=None,
    ):
        arg_hash = f"{version}{max_gradient}"
        arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
        self.name = f"{self.__class__.__name__}-{arg_hash}"
        self.version = str(version)
        assert self.version in self.VERSIONS
        self.max_gradient = max_gradient
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y", "neg_dy"),
        )

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1
        logging.info(f"Processing dataset {self.raw_file_names}")
        for energy, numbers, positions, forces in tqdm(
            parse_maceoff_tar(self.raw_paths[0]), desc="Processing conformations"
        ):
            data = Data(
                **dict(
                    z=pt.tensor(np.array(numbers), dtype=pt.long),
                    pos=pt.tensor(positions, dtype=pt.float32),
                    y=pt.tensor(energy, dtype=pt.float64).view(1, 1),
                    neg_dy=pt.tensor(forces, dtype=pt.float32),
                )
            )
            assert data.y.shape == (1, 1)
            assert data.z.shape[0] == data.pos.shape[0]
            assert data.neg_dy.shape[0] == data.pos.shape[0]
            # Skip samples with large forces
            if self.max_gradient:
                if data.neg_dy.norm(dim=1).max() > float(self.max_gradient):
                    continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            yield data

    def download(self):
        download_url(self.raw_url, self.raw_dir, filename=self.raw_file_names)
