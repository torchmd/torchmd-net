import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from .hdf import HDF5
import h5py

__all__ = ["Custom"]


def write_as_hdf5(coordfiles, embedfiles, energyfiles, forcefiles, hdf5_dataset):
    """Transform the input to hdf5 format compatible with the HDF5 Dataset class.
    The input files to the Custom dataset are transformed to a single HDF5 file.
    Args:
        coordfiles (list): List of coordinate files.
        embedfiles (list): List of embedding files.
        energyfiles (list): List of energy files.
        forcefiles (list): List of force files.
        hdf5_dataset (string): Path to the output HDF5 dataset.
    """
    with h5py.File(hdf5_dataset, "w") as f:
        for i in range(len(coordfiles)):
            # Create a group for each file
            coord_data = np.load(coordfiles[i])
            embed_data = np.load(embedfiles[i]).astype(int)
            group = f.create_group(str(i))
            num_samples = coord_data.shape[0]
            group["pos"] = coord_data
            group["types"] = np.tile(embed_data, (num_samples, 1))
            if energyfiles is not None:
                energy_data = np.load(energyfiles[i])
                group["energy"] = energy_data
            if forcefiles is not None:
                force_data = np.load(forcefiles[i])
                group["forces"] = force_data



class Custom(Dataset):
    r"""Custom Dataset to manage loading coordinates, embedding indices,
    energies and forces from NumPy files. :obj:`coordglob` and :obj:`embedglob`
    are required parameters. Either :obj:`energyglob`, :obj:`forceglob` or both
    must be given as targets.

    Args:
        coordglob (string): Glob path for coordinate files. Stored as "pos".
        embedglob (string): Glob path for embedding index files. Stored as "z" (atomic number).
        energyglob (string, optional): Glob path for energy files. Stored as "y".
            (default: :obj:`None`)
        forceglob (string, optional): Glob path for force files. Stored as "neg_dy".
            (default: :obj:`None`)
        preload_memory_limit (int, optional): If the dataset is smaller than this limit (in MB), preload it into CPU memory.
        read_as_hdf5 (string, optional): If present, transform the input files to HDF5 format and use the HDF5 Dataset.
    Example:
        >>> data = Custom(coordglob="coords_files*npy", embedglob="embed_files*npy")
        >>> sample = data[0]
        >>> assert hasattr(sample, "pos"), "Sample doesn't contain coords"
        >>> assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"

    Notes:
        For each sample in the data:
              - "pos" is an array of shape (n_atoms, 3)
              - "z" is an array of shape (n_atoms,).
              - If present, "y" is an array of shape (1,) and "neg_dy" has shape (n_atoms, 3)
    """

    def __init__(
        self,
        coordglob,
        embedglob,
        energyglob=None,
        forceglob=None,
        preload_memory_limit=1024,
        read_as_hdf5=None,
    ):
        super(Custom, self).__init__()
        assert energyglob is not None or forceglob is not None, (
            "Either energies, forces or both must " "be specified as the target"
        )
        self.has_energies = energyglob is not None
        self.has_forces = forceglob is not None

        self.coordfiles = sorted(glob.glob(coordglob))
        self.embedfiles = sorted(glob.glob(embedglob))
        self.energyfiles = sorted(glob.glob(energyglob)) if self.has_energies else None
        self.forcefiles = sorted(glob.glob(forceglob)) if self.has_forces else None

        assert len(self.coordfiles) == len(self.embedfiles), (
            f"Number of coordinate files {len(self.coordfiles)} "
            f"does not match number of embed files {len(self.embedfiles)}."
        )
        if self.has_energies:
            assert len(self.coordfiles) == len(self.energyfiles), (
                f"Number of coordinate files {len(self.coordfiles)} "
                f"does not match number of energy files {len(self.energyfiles)}."
            )
        if self.has_forces:
            assert len(self.coordfiles) == len(self.forcefiles), (
                f"Number of coordinate files {len(self.coordfiles)} "
                f"does not match number of force files {len(self.forcefiles)}."
            )
        print("Number of files: ", len(self.coordfiles))
        self.load_into_memory = False
        if read_as_hdf5 is not None:
            hdf5_file = read_as_hdf5
            if not self.has_energies:
                raise ValueError(
                    "HDF5 format requires energies to be present in the dataset."
                )
            write_as_hdf5(
                self.coordfiles,
                self.embedfiles,
                self.energyfiles,
                self.forcefiles,
                hdf5_file,
            )
            self.hdf5_dataset = HDF5(hdf5_file)
        total_data_size = self._initialize_index()
        print("Combined dataset size {}".format(len(self.index)))
        # If the dataset is small enough, load it whole into CPU memory
        data_size_limit = preload_memory_limit * 1024 * 1024
        if total_data_size < data_size_limit:
            self.load_into_memory = True
            print(
                "Preloading Custom dataset (of size {:.2f} MB) into CPU memory".format(
                    total_data_size / 1024 / 1024
                )
            )
            self._store_as_tensors()

    def _store_as_tensors(self):
        """Load the input files as Torch tensors.
        """
        self.coordfiles = torch.from_numpy(np.array([np.load(f) for f in self.coordfiles]))
        self.embedfiles = torch.from_numpy(
            np.array([np.load(f).astype(int) for f in self.embedfiles])
        )
        if self.has_energies:
            self.energyfiles = torch.from_numpy(np.array([np.load(f) for f in self.energyfiles]))
        if self.has_forces:
            self.forcefiles = torch.from_numpy(np.array([np.load(f) for f in self.forcefiles]))

    def _initialize_index(self):
        """Initialize the index for the dataset.
        Returns:
            int: Total size of the dataset in bytes.
        """
        # create index
        self.index = []
        nfiles = len(self.coordfiles)
        total_data_size = 0  # Number of bytes in the dataset
        for i in range(nfiles):
            coord_data = np.load(self.coordfiles[i])
            embed_data = np.load(self.embedfiles[i]).astype(int)
            size = coord_data.shape[0]
            total_data_size += coord_data.nbytes + embed_data.nbytes
            self.index.extend(list(zip([i] * size, range(size))))

            # consistency check
            assert coord_data.shape[1] == embed_data.shape[0], (
                f"Number of atoms in coordinate file {i} ({coord_data.shape[1]}) "
                f"does not match number of atoms in embed file {i} ({embed_data.shape[0]})."
            )
            if self.has_energies:
                energy_data = np.load(self.energyfiles[i])
                total_data_size += energy_data.nbytes
                assert coord_data.shape[0] == energy_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in energy file {i} ({energy_data.shape[0]})."
                )
            if self.has_forces:
                force_data = np.load(self.forcefiles[i])
                total_data_size += force_data.nbytes
                assert coord_data.shape == force_data.shape, (
                    f"Data shape of coordinate file {i} {coord_data.shape} "
                    f"does not match the shape of force file {i} {force_data.shape}."
                )
        return total_data_size

    def get(self, idx):
        if hasattr(self, "hdf5_dataset"):
            return self.hdf5_dataset.get(idx)
        fileid, index = self.index[idx]
        if self.load_into_memory:
            coord_data = self.coordfiles[fileid][index]
            embed_data = self.embedfiles[fileid]
        else:
            coord_data = torch.from_numpy(
                np.array(np.load(self.coordfiles[fileid], mmap_mode="r")[index])
            )
            embed_data = torch.from_numpy(np.load(self.embedfiles[fileid]).astype(int))

        features = dict(pos=coord_data, z=embed_data)

        if self.has_energies and self.load_into_memory:
            energy_data = self.energyfiles[fileid][index]
            features["y"] = energy_data
        elif self.has_energies:
            energy_data = torch.from_numpy(
                np.array(np.load(self.energyfiles[fileid], mmap_mode="r")[index])
            )

            features["y"] = energy_data

        if self.has_forces and self.load_into_memory:
            force_data = self.forcefiles[fileid][index]
            features["neg_dy"] = force_data
        elif self.has_forces:
            force_data = torch.from_numpy(
                np.array(np.load(self.forcefiles[fileid], mmap_mode="r")[index])
            )
            features["neg_dy"] = force_data

        return Data(**features)

    def len(self):
        return len(self.index)
