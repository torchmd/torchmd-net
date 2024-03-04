# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import numpy as np
import os
import torch as pt
from torch_geometric.data import Data, download_url
from torchmdnet.datasets.memdataset import MemmappedDataset
from torchmdnet.utils import ATOMIC_NUMBERS


class GenentechTorsions(MemmappedDataset):
    """Dataset of torsion scans of small molecules.

    This is a dataset consisting of torsion scans of small molecules.
    Gas-phase geometries and energies are calculated with CCSD(T)/CBS theory.
    By default we load the relative energies in the dataset which are relative to the minimum energy of the torsion scan.

    References:

    - https://pubs.acs.org/doi/10.1021/acs.jcim.6b00614
    """

    KCALMOL_TO_EV = 0.0433641153087705

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        paths=None,
        theory="CCSD_T_CBS_MP2",
        energy_field="deltaE",
    ):
        self.name = self.__class__.__name__
        self.paths = str(paths)
        self.theory = theory
        self.energy_field = energy_field
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            properties=("y"),
        )

    @property
    def raw_url(self):
        return "https://github.com/Acellera/sellers/raw/main/ci6b00614_si_002.zip"

    @property
    def raw_file_names(self):
        return [
            "QM_MM_Gas_Phase_Torsion_Scan_Individual_Results_with_CCSD_T_CBS_baseline.sdf"
        ]

    def download(self):
        import zipfile

        archive = download_url(self.raw_url, self.raw_dir)

        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(archive)

    def sample_iter(self, mol_ids=False):
        assert len(self.raw_paths) == 1

        with open(self.raw_paths[0]) as f:
            molstart_count = 0
            discard_molecule = False
            deltaE = None
            mol_id = None
            num_atoms = None
            scan_atoms = None
            z = []
            pos = []
            for line in f:
                if discard_molecule and not line.strip().startswith("$$$$"):
                    continue
                if molstart_count >= 0 and molstart_count < 4:
                    molstart_count += 1
                if molstart_count == 4:  # On the 4th line we read atom counts
                    num_atoms = int(line.strip().split()[0])
                    molstart_count = -1  # Start atom/bond section
                    continue
                if line.strip().startswith("$$$$"):
                    if not discard_molecule:
                        data = Data(
                            z=pt.tensor(z, dtype=pt.long),
                            pos=pt.tensor(np.vstack(pos), dtype=pt.float32),
                            y=pt.tensor(deltaE * self.KCALMOL_TO_EV, dtype=pt.float64),
                            mol_id=mol_id,
                            scan_atoms=scan_atoms,
                        )
                        yield data

                    molstart_count = 0
                    discard_molecule = False
                    deltaE = None
                    mol_id = None
                    num_atoms = None
                    scan_atoms = None
                    z = []
                    pos = []
                    continue

                # Parsing the atom section
                if num_atoms is not None:
                    num_atoms -= 1
                    if num_atoms >= 0:
                        pos_x, pos_y, pos_z, el = line.strip().split()[:4]
                        pos.append([float(pos_x), float(pos_y), float(pos_z)])
                        z.append(ATOMIC_NUMBERS[el])

                # Parsing the SDF properties
                if line.strip().startswith(">  <MinMethod>"):
                    min_method = next(f).strip()
                    if min_method != self.theory:
                        discard_molecule = True
                        continue
                if line.strip().startswith(f">  <{self.energy_field}>"):
                    deltaE = float(next(f).strip())
                if line.strip().startswith(">  <Number>"):
                    mol_id = int(next(f).strip())
                if line.strip().startswith(">  <ScanAtoms_1>"):
                    scan_atoms = map(int, next(f).strip().split())
