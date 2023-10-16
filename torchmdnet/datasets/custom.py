import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from .hdf import HDF5
import h5py

__all__ = ["Custom"]


def write_as_hdf5(files, hdf5_dataset):
    """Transform the input to hdf5 format compatible with the HDF5 Dataset class.
    The input files to the Custom dataset are transformed to a single HDF5 file.
    Args:
        files (dict): Dictionary of input files. Must contain "pos", "z" and at least one of "y" or "neg_dy".
        hdf5_dataset (string): Path to the output HDF5 dataset.
    """
    with h5py.File(hdf5_dataset, "w") as f:
        for i in range(len(files["pos"])):
            # Create a group for each file
            coord_data = np.load(files["pos"][i], mmap_mode="r")
            embed_data = np.load(files["z"][i], mmap_mode="r").astype(int)
            group = f.create_group(str(i))
            num_samples = coord_data.shape[0]
            group["pos"] = coord_data
            group["types"] = np.tile(embed_data, (num_samples, 1))
            if "y" in files:
                energy_data = np.load(files["y"][i], mmap_mode="r")
                group["energy"] = energy_data
            if "neg_dy" in files:
                force_data = np.load(files["neg_dy"][i], mmap_mode="r")
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
        self.fields = [
            ("pos", "pos", torch.float32),
            ("z", "types", torch.long),
        ]
        self.files = {}
        self.files["pos"] = sorted(glob.glob(coordglob))
        self.files["z"] = sorted(glob.glob(embedglob))
        assert len(self.files["pos"]) == len(self.files["z"]), (
            f"Number of coordinate files {len(self.files['pos'])} "
            f"does not match number of embed files {len(self.files['z'])}."
        )
        if self.has_energies:
            self.files["y"] = sorted(glob.glob(energyglob))
            self.fields.append(("y", "energy", torch.float32))
            assert len(self.files["pos"]) == len(self.files["y"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of energy files {len(self.files['y'])}."
            )
        if self.has_forces:
            self.files["neg_dy"] = sorted(glob.glob(forceglob))
            self.fields.append(("neg_dy", "forces", torch.float32))
            assert len(self.files["pos"]) == len(self.files["neg_dy"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of force files {len(self.files['neg_dy'])}."
            )
        print("Number of files: ", len(self.files["pos"]))
        self.cached = False
        if read_as_hdf5 is not None:
            hdf5_file = read_as_hdf5
            write_as_hdf5(
                self.files,
                hdf5_file,
            )
            self.hdf5_dataset = HDF5(hdf5_file)
        total_data_size = self._initialize_index()
        print("Combined dataset size {}".format(len(self.index)))
        # If the dataset is small enough, load it whole into CPU memory
        data_size_limit = preload_memory_limit * 1024 * 1024
        if total_data_size < data_size_limit:
            self.cached = True
            print(
                "Preloading Custom dataset (of size {:.2f} MB)".format(
                    total_data_size / 1024**2
                )
            )
            self._preload_data()
        else:
            self._store_numpy_memmaps()

    def _preload_data(self):
        """Load the input files as Torch tensors.
        Each file can have different number of atoms, so each one is stored in a different tensor.
        """
        self.stored_data = {}
        self.stored_data["pos"] = [
            torch.from_numpy(np.load(f)) for f in self.files["pos"]
        ]
        self.stored_data["z"] = [
            torch.from_numpy(np.load(f).astype(int))
            .unsqueeze(0)
            .expand(self.stored_data["pos"][i].shape[0], -1)
            for i, f in enumerate(self.files["z"])
        ]
        if self.has_energies:
            self.stored_data["y"] = [
                torch.from_numpy(np.load(f)) for f in self.files["y"]
            ]
        if self.has_forces:
            self.stored_data["neg_dy"] = [
                torch.from_numpy(np.load(f)) for f in self.files["neg_dy"]
            ]

    def _store_numpy_memmaps(self):
        """Create a dictionary with numpy arrays with mmap_mode="r" for each file."""
        self.stored_data = {}
        self.stored_data["pos"] = [np.load(f, mmap_mode="r") for f in self.files["pos"]]
        self.stored_data["z"] = []
        for i, f in enumerate(self.files["z"]):
            loaded_data = np.load(f).astype(int)
            desired_shape = (len(self.stored_data["pos"][i]), loaded_data.shape[0])
            broadcasted_data = np.broadcast_to(
                loaded_data[np.newaxis, :], desired_shape
            )
            self.stored_data["z"].append(broadcasted_data)
        if self.has_energies:
            self.stored_data["y"] = [np.load(f, mmap_mode="r") for f in self.files["y"]]
        if self.has_forces:
            self.stored_data["neg_dy"] = [
                np.load(f, mmap_mode="r") for f in self.files["neg_dy"]
            ]

    def _initialize_index(self):
        """Initialize the index for the dataset.
        The index relates a sample to the file it belongs to and the index within that file.
        Returns:
            int: Total size of the dataset in bytes.
        """
        # create index
        self.index = []
        nfiles = len(self.files["pos"])
        total_data_size = 0  # Number of bytes in the dataset
        for i in range(nfiles):
            coord_data = np.load(self.files["pos"][i], mmap_mode="r")
            embed_data = np.load(self.files["z"][i]).astype(int)
            size = coord_data.shape[0]
            total_data_size += coord_data.nbytes + embed_data.nbytes
            self.index.extend(list(zip([i] * size, range(size))))
            # consistency check
            assert coord_data.shape[1] == embed_data.shape[0], (
                f"Number of atoms in coordinate file {i} ({coord_data.shape[1]}) "
                f"does not match number of atoms in embed file {i} ({embed_data.shape[0]})."
            )
            if self.has_energies:
                energy_data = np.load(self.files["y"][i], mmap_mode="r")
                total_data_size += energy_data.nbytes
                assert coord_data.shape[0] == energy_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in energy file {i} ({energy_data.shape[0]})."
                )
            if self.has_forces:
                force_data = np.load(self.files["neg_dy"][i], mmap_mode="r")
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
        data = Data()
        for field in self.fields:
            # The dataset is stored as mem mapped numpy arrays unless it is cached,
            # in which case it is already stored as torch tensors
            f = self.stored_data[field[0]][fileid][index]
            data[field[0]] = f if self.cached else torch.from_numpy(np.array(f))
        return data

    def len(self):
        return len(self.index)
