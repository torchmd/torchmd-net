# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import os
import h5py
import torch
import math
import logging
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from torch_geometric.data import Dataset, Data
import urllib.request
from collections import defaultdict

logger = logging.getLogger("MDCATH")


def load_pdb_list(pdb_list):
    """Load PDB list from a file or return list directly."""
    if isinstance(pdb_list, list):
        return pdb_list
    elif isinstance(pdb_list, str) and os.path.isfile(pdb_list):
        logger.info(f"Reading PDB list from {pdb_list}")
        with open(pdb_list, "r") as file:
            return [line.strip() for line in file]
    raise ValueError("Invalid PDB list. Please provide a list or a path to a file.")


class MDCATH(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        source_file="mdcath_source.h5",
        file_basename="mdcath_dataset",
        numAtoms=5000,
        numNoHAtoms=None,
        numResidues=1000,
        temperatures=["348"],
        skip_frames=1,
        pdb_list=None,
        min_gyration_radius=None,
        max_gyration_radius=None,
        alpha_beta_coil=None,
        solid_ss=None,
        numFrames=None,
    ):
        """mdCATH dataset class for PyTorch Geometric to load protein structures and dynamics from the mdCATH dataset.

        Parameters:
        -----------
        root: str
            Root directory where the dataset should be stored. Data will be downloaded to 'root/'.
        numAtoms: int
            Max number of atoms in the protein structure.
        source_file: str
            Name of the source file with the information about the protein structures. Default is "mdcath_source.h5".
        file_basename: str
            Base name of the hdf5 files. Default is "mdcath_dataset".
        numNoHAtoms: int
            Max number of non-hydrogen atoms in the protein structure, not available for original mdcath dataset. Default is None.
            Be sure to have the attribute 'numNoHAtoms' in the source file.
        numResidues: int
            Max number of residues in the protein structure.
        temperatures: list
            List of temperatures (in Kelvin) to download. Default is ["348"]. Available temperatures are ['320', '348', '379', '413', '450']
        skip_frames: int
            Number of frames to skip in the trajectory. Default is 1.
        pdb_list: list or str
            List of PDB IDs to download or path to a file with the PDB IDs. If None, all available PDB IDs from 'source_file' will be downloaded.
            The filters will be applied to the PDB IDs in this list in any case. Default is None.
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

        self.url = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/"
        self.source_file = source_file
        self.file_basename = file_basename
        self.numNoHAtoms = numNoHAtoms
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.numAtoms = numAtoms
        self.numResidues = numResidues
        self.temperatures = [str(temp) for temp in temperatures]
        self.skip_frames = skip_frames
        self.pdb_list = load_pdb_list(pdb_list) if pdb_list is not None else None
        self.min_gyration_radius = min_gyration_radius
        self.max_gyration_radius = max_gyration_radius
        self.alpha_beta_coil = alpha_beta_coil
        self.numFrames = numFrames
        self.solid_ss = solid_ss
        self._ensure_source_file()
        self._filter_and_prepare_data()
        self.idx = None
        super(MDCATH, self).__init__(root, transform, pre_transform, pre_filter)
        # Calculate the total size of the dataset in MB
        self.total_size_mb = self.calculate_dataset_size()

        logger.info(f"Total number of domains: {len(self.processed.keys())}")
        logger.info(f"Total number of conformers: {self.num_conformers}")
        logger.info(f"Total size of dataset: {self.total_size_mb} MB")

    @property
    def raw_file_names(self):
        return [f"{self.file_basename}_{pdb_id}.h5" for pdb_id in self.processed.keys()]

    @property
    def raw_dir(self):
        # Override the raw_dir property to return the root directory
        # The files will be downloaded to the root directory, compatible only with original mdcath dataset
        return self.root

    def _ensure_source_file(self):
        """Ensure the source file is downloaded before processing."""
        source_path = os.path.join(self.root, self.source_file)
        if not os.path.exists(source_path):
            assert (
                self.source_file == "mdcath_source.h5"
            ), "Only 'mdcath_source.h5' is supported as source file for download."
            logger.info(f"Downloading source file {self.source_file}")
            urllib.request.urlretrieve(opj(self.url, self.source_file), source_path)

    def download(self):
        for pdb_id in self.processed.keys():
            file_name = f"{self.file_basename}_{pdb_id}.h5"
            file_path = opj(self.raw_dir, file_name)
            if not os.path.exists(file_path):
                assert (
                    self.file_basename == "mdcath_dataset"
                ), "Only 'mdcath_dataset' is supported as file_basename for download."
                # Download the file if it does not exist
                urllib.request.urlretrieve(opj(self.url, "data", file_name), file_path)

    def calculate_dataset_size(self):
        total_size_bytes = 0
        for pdb_id in self.processed.keys():
            file_name = f"{self.file_basename}_{pdb_id}.h5"
            total_size_bytes += os.path.getsize(opj(self.root, file_name))
        total_size_mb = round(total_size_bytes / (1024 * 1024), 4)
        return total_size_mb

    def _filter_and_prepare_data(self):
        source_info_path = os.path.join(self.root, self.source_file)

        self.processed = defaultdict(list)
        self.num_conformers = 0

        with h5py.File(source_info_path, "r") as file:
            domains = file.keys() if self.pdb_list is None else self.pdb_list

            for pdb_id in tqdm(domains, desc="Processing mdcath source"):
                pdb_group = file[pdb_id]
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
                self._process_temperatures(pdb_id, pdb_group)

    def _process_temperatures(self, pdb_id, pdb_group):
        for temp in self.temperatures:
            for replica in pdb_group[temp].keys():
                self._evaluate_replica(pdb_id, temp, replica, pdb_group)

    def _evaluate_replica(self, pdb_id, temp, replica, pdb_group):
        conditions = [
            self.numFrames is not None
            and pdb_group[temp][replica].attrs["numFrames"] < self.numFrames,
            self.min_gyration_radius is not None
            and pdb_group[temp][replica].attrs["min_gyration_radius"]
            < self.min_gyration_radius,
            self.max_gyration_radius is not None
            and pdb_group[temp][replica].attrs["max_gyration_radius"]
            > self.max_gyration_radius,
            self._evaluate_structure(pdb_group, temp, replica),
            self.numNoHAtoms is not None
            and pdb_group.attrs["numNoHAtoms"] > self.numNoHAtoms,
        ]
        if any(conditions):
            return

        num_frames = math.ceil(
            pdb_group[temp][replica].attrs["numFrames"] / self.skip_frames
        )
        self.processed[pdb_id].append((temp, replica, num_frames))
        self.num_conformers += num_frames

    def _evaluate_structure(self, pdb_group, temp, replica):
        alpha = pdb_group[temp][replica].attrs["alpha"]
        beta = pdb_group[temp][replica].attrs["beta"]
        solid_ss = (alpha + beta) / pdb_group.attrs["numResidues"] * 100
        return self.solid_ss is not None and solid_ss < self.solid_ss

    def len(self):
        return self.num_conformers

    def _setup_idx(self):
        files = [
            opj(self.root, f"{self.file_basename}_{pdb_id}.h5")
            for pdb_id in self.processed.keys()
        ]
        self.idx = []
        for i, (pdb, group_info) in enumerate(self.processed.items()):
            for temp, replica, num_frames in group_info:
                # build the catalog here for each conformer
                d = [
                    (pdb, files[i], temp, replica, conf_id)
                    for conf_id in range(num_frames)
                ]
                self.idx.extend(d)

        assert (
            len(self.idx) == self.num_conformers
        ), f"Mismatch between number of conformers and idxs: {self.num_conformers} vs {len(self.idx)}"

    def process_specific_group(self, pdb, file, temp, repl, conf_idx):
        # do not use attributes from h5group because is will cause memory leak
        # use the read_direct and np.s_ to get the coords and forces of interest directly
        conf_idx = conf_idx * self.skip_frames
        slice_idxs = np.s_[conf_idx : conf_idx + 1]
        with h5py.File(file, "r") as f:
            z = f[pdb]["z"][:]
            coords = np.zeros((z.shape[0], 3))
            forces = np.zeros((z.shape[0], 3))

            group = f[f"{pdb}/{temp}/{repl}"]

            group["coords"].read_direct(coords, slice_idxs)
            group["forces"].read_direct(forces, slice_idxs)

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
        z, coords, forces = self.process_specific_group(
            pdb_id, file_path, temp, replica, conf_idx
        )
        data.z = torch.tensor(z, dtype=torch.long)
        data.pos = torch.tensor(coords, dtype=torch.float)
        data.neg_dy = torch.tensor(forces, dtype=torch.float)
        data.info = f"{pdb_id}_{temp}_{replica}_{conf_idx}"
        return data
