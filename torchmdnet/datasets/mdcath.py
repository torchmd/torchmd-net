# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import os
from os.path import join as opj
import h5py
import torch
from tqdm import tqdm
import math
import numpy as np
from torch_geometric.data import Dataset, download_url, Data


def get_pdb_list(pdb_list):
    # pdb list could be a list of pdb ids or a file with the pdb ids
    if isinstance(pdb_list, list):
        return pdb_list
    elif isinstance(pdb_list, str):
        if os.path.exists(pdb_list):
            print(f"Reading PDB list from {pdb_list}")
            with open(pdb_list, "r") as f:
                return [line.strip() for line in f]
        else:
            raise FileNotFoundError(f"File {pdb_list} not found")
    else:
        return None


class mdCATH(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preload_dataset_limit=None,
        numAtoms=5000,
        numNoHAtoms=None,
        numResidues=1000,
        temperatures=["348"],
        skipFrames=1,
        pdb_list=None,
        min_gyration_radius=None,
        max_gyration_radius=None,
        alpha_beta_coil=None,
        solid_ss = None,
        numFrames=None,
    ):
        """mdCATH dataset class for PyTorch Geometric to load protein structures and dynamics from the mdCATH dataset.

        Parameters:
        -----------
        root: str
            Root directory where the dataset should be stored. Data will be downloaded to 'root/'.
        numAtoms: int
            Max number of atoms in the protein structure.
        numNoHAtoms: int
            Max number of non-hydrogen atoms in the protein structure.
        numResidues: int
            Max number of residues in the protein structure.
        temperatures: list
            List of temperatures (in Kelvin) to download. Default is ["348"]. Available temperatures are ['320', '348', '379', '413', '450']
        skipFrames: int
            Number of frames to skip in the trajectory. Default is 1.
        pdb_list: list or str
            List of PDB IDs to download. If None, all available PDB IDs from 'mdcath_source.h5' will be downloaded.
        min_gyration_radius: float
            Minimum gyration radius (in nm) of the protein structure. Default is None.
        max_gyration_radius: float
            Maximum gyration radius (in nm) of the protein structure. Default is None.
        alpha_beta_coil: tuple
            Tuple with the minimum percentage of alpha-helix, beta-sheet and coil residues in the protein structure. Default is None.
        solid_ss: float
            minimum percentage of solid secondary structure in the protein structure (alpha + beta)/total_residues * 100. Default is None.
        numFrames: int
            Minimum number of frames in the trajectory in order to be considered. Default is None.
        """

        self.url = "https://zenodo.org/record/<record_id>/files/"
        self.preload_dataset_limit = preload_dataset_limit
        super(mdCATH, self).__init__(root, transform, pre_transform, pre_filter)

        self.numAtoms = numAtoms
        self.numNoHAtoms = numNoHAtoms
        self.numResidues = numResidues
        self.temperatures = temperatures
        self.skipFrames = skipFrames
        self.pdb_list = get_pdb_list(pdb_list)
        self.min_gyration_radius = min_gyration_radius
        self.max_gyration_radius = max_gyration_radius
        self.alpha_beta_coil = alpha_beta_coil
        self.numFrames = numFrames
        self.solid_ss = solid_ss
        self.idx = None
        self.process_data_source()
        # Calculate the total size of the dataset in MB
        self.total_size_mb = self.calculate_dataset_size()
        
        print(f"Total number of domains: {len(self.to_download.keys())}")
        print(f"Total number of conformers: {self.num_conformers}")
        print(f"Total size of dataset: {self.total_size_mb} MB")

    @property
    def raw_file_names(self):
        # Check if the dataset has been processed, and if not, return the original source file
        if not hasattr(self, "to_download"):
            return ["mdCATH_source.h5"]
        # Otherwise, return the list of HDF5 files that passed the filtering criteria
        return [f"cath_dataset_{pdb_id}.h5" for pdb_id in self.to_download.keys()]

    @property
    def raw_dir(self):
        # Override the raw_dir property to prevent the creation of a 'raw' directory
        # The files will be downloaded to the root directory
        return self.root
    def download(self):
        if not hasattr(self, "to_download") or not self.to_download:
            download_url(opj(self.url, "mdCATH_source.h5"), self.root)
            return
        for pdb_id in self.to_download.keys():
    def calculate_dataset_size(self):
        total_size_bytes = 0
        for pdb_id in self.to_download.keys():
            file_name = f"cath_noh_dataset_{pdb_id}.h5" if self.noh_mode else f"cath_dataset_{pdb_id}.h5"
            total_size_bytes += os.path.getsize(opj(self.root, file_name))
        total_size_mb = round(total_size_bytes / (1024 * 1024), 4)
        return total_size_mb
    def process_data_source(self):
        print("Processing mdCATH source")
        data_info_path = opj(self.root, "mdCATH_source.h5")
        if not os.path.exists(data_info_path):
            self.download()
        # the to_downlaod is the dictionary that will store the pdb ids and the corresponding temp and replica ids if they pass the filter
        self.to_download = {}
        self.num_conformers = 0
        with h5py.File(data_info_path, "r") as f:
            domains = f.keys() if self.pdb_list is None else self.pdb_list
            for pdb in tqdm(
                domains, total=len(domains), desc="Processing mdCATH source"
            ):
                pdb_group = f[pdb]
                if (
                    self.numAtoms is not None
                    and pdb_group.attrs["numProteinAtoms"] > self.numAtoms
                ):
                    continue
                if (
                    self.numResidues is not None
                    and pdb_group.attrs["numResidues"] > self.numResidues
                ):
                    continue
                if (
                    self.numNoHAtoms is not None
                    and pdb_group.attrs["numNoHAtoms"] > self.numNoHAtoms
                ):
                    continue
                for temp in self.temperatures:
                    if temp not in pdb_group.keys():
                        continue
                    for replica in pdb_group[temp].keys():
                        if (
                            self.numFrames is not None
                            and pdb_group[temp][replica].attrs["numFrames"]
                            < self.numFrames
                        ):
                            continue
                        if (
                            self.min_gyration_radius is not None
                            and pdb_group[temp][replica].attrs["min_gyration_radius"]
                            < self.min_gyration_radius
                        ):
                            continue
                        if (
                            self.max_gyration_radius is not None
                            and pdb_group[temp][replica].attrs["max_gyration_radius"]
                            > self.max_gyration_radius
                        ):
                            continue
                        if (
                            self.alpha_beta_coil is not None
                            or self.solid_ss is not None
                        ):
                            alpha = pdb_group[temp][replica].attrs["alpha"]
                            beta = pdb_group[temp][replica].attrs["beta"]
                            coil = pdb_group[temp][replica].attrs["coil"]
                            solid_ss = (
                                (alpha + beta) / pdb_group.attrs["numResidues"] * 100
                            )
                            if self.solid_ss is not None:
                                if solid_ss < self.solid_ss:
                                    continue
                            else:
                                if not np.isclose(
                                    [alpha, beta, coil], list(self.alpha_beta_coil)
                                ).all():
                                    continue
                        if pdb not in self.to_download:
                            self.to_download[pdb] = []
                        self.to_download[pdb].append((temp, replica))
                        # append the number of frames of the trajectory to the total number of molecules
                        self.num_conformers += math.ceil(
                            pdb_group[temp][replica].attrs["numFrames"]
                            / self.skipFrames
                        )

    def len(self):
        return self.num_conformers

    def process_specific_group(self, pdb, file, group_info):
        with h5py.File(file, "r") as f:
            z = f[pdb]["z"][()]
            group = f[pdb][f"sims{group_info[0]}K"][group_info[1]]
            coords = group["coords"][()][:: self.skipFrames, :, :]
            forces = group["forces"][()][:: self.skipFrames, :, :]
            # coords and forces shape (num_frames, num_atoms, 3)

            assert (
                coords.shape[0] == forces.shape[0]
            ), f"Number of frames mismatch between coords and forces: {group['coords'].shape[0]} vs {group['forces'].shape[0]}"
            assert (
                coords.shape[1] == z.shape[0]
            ), f"Number of atoms mismatch between coords and z: {group['coords'].shape[1]} vs {z.shape[0]}"
            assert (
                forces.shape[1] == z.shape[0]
            ), f"Number of atoms mismatch between forces and z: {group['forces'].shape[1]} vs {z.shape[0]}"
            assert (
                group["coords"].attrs["unit"] == "Angstrom"
            ), f"Coords unit is not Angstrom: {group['coords'].attrs['unit']}"
            assert (
                group["forces"].attrs["unit"] == "kcal/mol/Angstrom"
            ), f"Forces unit is not kcal/mol/Angstrom: {group['forces'].attrs['unit']}"

            return [z, coords, forces]

    def _setup_idx(self):
        files = [
            opj(self.root, f"cath_dataset_{pdb_id}.h5")
            for pdb_id in self.to_download.keys()
        ]
        self.idx = []
        for i, (pdb, group_info) in enumerate(self.to_download.items()):
            for temp, replica in group_info:
                data = self.process_specific_group(pdb, files[i], (temp, replica))
                conformer_indices = range(data[1].shape[0])
                self.idx.extend(
                    [
                        tuple([data[0], data[1][j], data[2][j], [j]])
                        for j in conformer_indices
                    ]
                )
        assert (
            len(self.idx) == self.num_conformers
        ), f"Mismatch between number of conformers and idxs: {self.num_conformers} vs {len(self.idx)}"

    def get(self, idx):
        data = Data()
        if self.idx is None:
            self._setup_idx()
        *fields_data, i = self.idx[idx]
        data.z = torch.tensor(fields_data[0][i], dtype=torch.long)
        data.pos = torch.tensor(fields_data[1][i], dtype=torch.float32)
        data.neg_dy = torch.tensor(fields_data[2][i], dtype=torch.float32)
        return data
