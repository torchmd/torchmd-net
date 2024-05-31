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
        preload_dataset_limit: int
            Maximum size of the dataset in MB to load into memory. If the dataset is larger than this limit, a warning will be printed. Default is 1024 MB.
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
        self.source_file = "mdcath_source.h5"
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
            return [self.source_file]
        # Otherwise, return the list of HDF5 files that passed the filtering criteria
        return [f"mdcath_dataset_{pdb_id}.h5" for pdb_id in self.to_download.keys()]

    @property
    def raw_dir(self):
        # Override the raw_dir property to prevent the creation of a 'raw' directory
        # The files will be downloaded to the root directory
        return self.root
    def download(self):
        if not hasattr(self, "to_download") or not self.to_download:
            download_url(opj(self.url, self.source_file), self.root)
            return
        for pdb_id in self.to_download.keys():
            file_name = f"mdcath_dataset_{pdb_id}.h5"
            file_path = opj(self.raw_dir, file_name)
            if not os.path.exists(file_path):
                download_url(opj(self.url, file_name), self.root)

    def calculate_dataset_size(self):
        total_size_bytes = 0
        for pdb_id in self.to_download.keys():
            file_name = f"mdcath_dataset_{pdb_id}.h5"
            total_size_bytes += os.path.getsize(opj(self.root, file_name))
        total_size_mb = round(total_size_bytes / (1024 * 1024), 4)
        return total_size_mb
    def process_data_source(self):
        print("Processing mdCATH source")
        data_info_path = opj(self.root, self.source_file)
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
                            num_frames = math.ceil(pdb_group[temp][replica].attrs["numFrames"]/self.skipFrames)
                            self.to_download[pdb].append((temp, replica, num_frames))
                            # append the number of frames of the trajectory to the total number of molecules
                            self.num_conformers += num_frames

    def len(self):
        return self.num_conformers

    def _setup_idx(self):
        files = [opj(self.root, f"mdcath_dataset_{pdb_id}.h5") for pdb_id in self.to_download.keys()]
        self.idx = []
        for i, (pdb, group_info) in enumerate(self.to_download.items()):
            for temp, replica, num_frames in group_info:
                # build the catalog here for each conformer
                d = [(pdb, files[i], temp, replica, conf_id) for conf_id in range(num_frames)]
                self.idx.extend(d)  
                
        assert (len(self.idx) == self.num_conformers), f"Mismatch between number of conformers and idxs: {self.num_conformers} vs {len(self.idx)}"
    
    
    def process_specific_group(self, pdb, file, temp, repl, conf_idx):
        # do not use attributes from h5group beause is will cause memory leak
        # use the read_direct and np.s_ to get the coords and forces of interest directly
        conf_idx = conf_idx*self.skipFrames 
        slice_idxs = np.s_[conf_idx:conf_idx+1]
        with h5py.File(file, "r") as f:
            z = f[pdb]["z"][:]
            coords = np.zeros((z.shape[0], 3))
            forces = np.zeros((z.shape[0], 3))
            
            group = f[f'{pdb}/{temp}/{repl}']

            group['coords'].read_direct(coords, slice_idxs)
            group['forces'].read_direct(forces, slice_idxs) 
                         
            # coords and forces shape (num_atoms, 3)
            assert (
                coords.shape[0] == forces.shape[0]
            ), f"Number of frames mismatch between coords and forces: {group['coords'].shape[0]} vs {group['forces'].shape[0]}"
            assert (
                coords.shape[0] == z.shape[0]
            ), f"Number of atoms mismatch between coords and z: {group['coords'].shape[1]} vs {z.shape[0]}"
        return (z, coords, forces)
    
    def get(self, element):
        data = Data()
        if self.idx is None:
            # this process will be performed, num_workers * num_gpus (one per thread)
            self._setup_idx()
        # fields_data is a tuple with the file, pdb, temp, replica, conf_idx
        pdb_id, file_path, temp, replica, conf_idx = self.idx[element]
        z, coords, forces = self.process_specific_group(pdb_id, file_path, temp, replica, conf_idx)
        data.z = torch.tensor(z, dtype=torch.long)
        data.pos = torch.tensor(coords, dtype=torch.float)
        data.neg_dy = torch.tensor(forces, dtype=torch.float)
        return data