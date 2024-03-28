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
        numFrames=None,
    ):
        """ mdCATH dataset class for PyTorch Geometric to load protein structures and dynamics from the mdCATH dataset.
        
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
        pdb_list: list
            List of PDB IDs to download. If None, all available PDB IDs from 'mdcath_source.h5' will be downloaded.
        min_gyration_radius: float
            Minimum gyration radius (in nm) of the protein structure. Default is None.
        max_gyration_radius: float
            Maximum gyration radius (in nm) of the protein structure. Default is None.
        alpha_beta_coil: tuple
            Tuple with the minimum percentage of alpha-helix, beta-sheet and coil residues in the protein structure. Default is None.
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
        self.pdb_list = pdb_list
        self.min_gyration_radius = min_gyration_radius
        self.max_gyration_radius = max_gyration_radius
        self.alpha_beta_coil = alpha_beta_coil
        self.numFrames = numFrames
        self.idx = None
        self.process_data_source()
        print(f"Total number of domains: {len(self.to_download.keys())}")
        print(f"Total number of conformers: {self.num_conformers}")
        
    @property
    def raw_file_names(self):
        # Check if the dataset has been processed, and if not, return the original source file
        if not hasattr(self, 'to_download'):
            return ['mdCATH_source.h5']
        # Otherwise, return the list of HDF5 files that passed the filtering criteria
        return [f"cath_dataset_{pdb_id}.h5" for pdb_id in self.to_download.keys()]

    def download(self):
        if not hasattr(self, 'to_download') or not self.to_download:
            download_url(opj(self.url, 'mdCATH_source.h5'), self.root)
            return          
        for pdb_id in self.to_download.keys():
            download_url(opj(self.url, f"cath_dataset_{pdb_id}.h5"), self.root)
    
    def process_data_source(self):
        print("Processing mdCATH source")
        data_info_path = opj(self.root, 'mdCATH_source.h5')
        if not os.path.exists(data_info_path):
            self.download()
        # the to_downlaod is the dictionary that will store the pdb ids and the corresponding temp and replica ids if they pass the filter
        self.to_download = {}
        self.num_conformers = 0
        with h5py.File(data_info_path, 'r') as f:
            domains = f.keys() if self.pdb_list is None else self.pdb_list
            for pdb in tqdm(domains, total=len(domains), desc="Processing mdCATH source"):
                pdb_group = f[pdb]
                if pdb_group.attrs['numProteinAtoms'] > self.numAtoms:
                    continue
                if pdb_group.attrs['numResidues'] > self.numResidues:
                    continue
                if self.numNoHAtoms is not None and pdb_group.attrs['numNoHAtoms'] > self.numNoHAtoms:
                    continue
                for temp in self.temperatures:
                    if temp not in pdb_group.keys():
                        continue
                for replica in pdb_group[temp].keys():
                    if self.numFrames is not None and pdb_group[temp][replica].attrs['numFrames'] < self.numFrames:
                        continue
                    if self.min_gyration_radius is not None and pdb_group[temp][replica].attrs['min_gyration_radius'] < self.min_gyration_radius:
                        continue
                    if self.max_gyration_radius is not None and pdb_group[temp][replica].attrs['max_gyration_radius'] > self.max_gyration_radius:
                        continue
                    if self.alpha_beta_coil is not None:
                        alpha = pdb_group[temp][replica].attrs['alpha']
                        beta = pdb_group[temp][replica].attrs['beta']
                        coil = pdb_group[temp][replica].attrs['coil']
                        if not np.isclose([alpha, beta, coil], list(self.alpha_beta_coil)).all():
                            continue
                    if pdb not in self.to_download:
                        self.to_download[pdb] = []
                    self.to_download[pdb].append((temp, replica))
                    # append the number of frames of the trajectory to the total number of molecules
                    self.num_conformers += math.ceil(pdb_group[temp][replica].attrs['numFrames'] / self.skipFrames)
                    
    def len(self):
        return self.num_conformers
    
    def process_specific_group(self, pdb, file, group_info):
        with h5py.File(file, 'r') as f:
            z = f[pdb]['z'][()]
            group = f[pdb][f'sims{group_info[0]}K'][group_info[1]]
            # if any assertion fail print the same message 
            # coords and forces shape (num_frames, num_atoms, 3)
            
            assert group['coords'].shape[0] == group['forces'].shape[0], f"Number of frames mismatch between coords and forces: {group['coords'].shape[0]} vs {group['forces'].shape[0]}"
            assert group['coords'].shape[1] == z.shape[0], f"Number of atoms mismatch between coords and z: {group['coords'].shape[1]} vs {z.shape[0]}"
            assert group['forces'].shape[1] == z.shape[0], f"Number of atoms mismatch between forces and z: {group['forces'].shape[1]} vs {z.shape[0]}"
            assert group['coords'].attrs['unit'] == 'Angstrom', f"Coords unit is not Angstrom: {group['coords'].attrs['unit']}"
            assert group['forces'].attrs['unit'] == 'kcal/mol/Angstrom', f"Forces unit is not kcal/mol/Angstrom: {group['forces'].attrs['unit']}"
            
            coords = torch.tensor(group['coords'][()])[::self.skipFrames, :, :]
            forces = torch.tensor(group['forces'][()])[::self.skipFrames, :, :]
            z = torch.tensor(z)
            return Data(pos=coords, neg_dy=forces, z=z)
    
    def _setup_idx(self):
        files = [opj(self.root,f"cath_dataset_{pdb_id}.h5") for pdb_id in self.to_download.keys()]
        self.idx = []
        for i, (pdb, group_info) in enumerate(self.to_download.items()):
            for temp, replica in group_info:
                data = self.process_specific_group(pdb, files[i], (temp, replica))
                self.idx.extend([(data.pos[frame], data.neg_dy[frame], data.z) for frame in range(data.pos.shape[0])])
        assert len(self.idx) == self.num_conformers, f"Mismatch between number of conformers and idxs: {self.num_conformers} vs {len(self.idx)}"
    
    def get(self, idx):
        data = Data()
        if self.idx is None:
            self._setup_idx()
        data.pos = self.idx[idx][0]
        data.neg_dy = self.idx[idx][1]
        data.z = self.idx[idx][2]
        return data