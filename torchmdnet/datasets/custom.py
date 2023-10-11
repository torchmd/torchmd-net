import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


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
        load_into_memory_limit (int, optional): If the dataset is smaller than this limit (in MB), preload it into CPU memory.
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
        print("Combined dataset size {}".format(len(self.index)))
        # If the dataset is small enough, load it whole into CPU memory
        data_size_limit = preload_memory_limit * 1024 * 1024
        self.load_into_memory = False
        if total_data_size < data_size_limit:
            self.load_into_memory = True
            print(
                "Preloading Custom dataset (of size {:.2f} MB) into CPU memory".format(
                    total_data_size / 1024 / 1024
                )
            )
            self.coordfiles = torch.from_numpy(
                np.array([np.load(f) for f in self.coordfiles])
            )
            self.embedfiles = torch.from_numpy(
                np.array([np.load(f).astype(int) for f in self.embedfiles])
            )
            if self.has_energies:
                self.energyfiles = torch.from_numpy(
                    np.array([np.load(f) for f in self.energyfiles])
                )
            if self.has_forces:
                self.forcefiles = torch.from_numpy(
                    np.array([np.load(f) for f in self.forcefiles])
                )

    def get(self, idx):
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
