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
        coordglob (string): Glob path for coordinate files.
        embedglob (string): Glob path for embedding index files.
        energyglob (string, optional): Glob path for energy files.
            (default: :obj:`None`)
        forceglob (string, optional): Glob path for force files.
            (default: :obj:`None`)
    """

    def __init__(self, coordglob, embedglob, energyglob=None, forceglob=None):
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
        for i in range(nfiles):
            coord_data = np.load(self.coordfiles[i])
            embed_data = np.load(self.embedfiles[i]).astype(np.int)
            size = coord_data.shape[0]
            self.index.extend(list(zip([i] * size, range(size))))

            # consistency check
            assert coord_data.shape[1] == embed_data.shape[0], (
                f"Number of atoms in coordinate file {i} ({coord_data.shape[1]}) "
                f"does not match number of atoms in embed file {i} ({embed_data.shape[0]})."
            )
            if self.has_energies:
                energy_data = np.load(self.energyfiles[i])
                assert coord_data.shape[0] == energy_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in energy file {i} ({energy_data.shape[0]})."
                )
            if self.has_forces:
                force_data = np.load(self.forcefiles[i])
                assert coord_data.shape == force_data.shape, (
                    f"Data shape of coordinate file {i} {coord_data.shape} "
                    f"does not match the shape of force file {i} {force_data.shape}."
                )
        print("Combined dataset size {}".format(len(self.index)))

    def get(self, idx):
        fileid, index = self.index[idx]

        coord_data = np.array(np.load(self.coordfiles[fileid], mmap_mode="r")[index])
        embed_data = np.load(self.embedfiles[fileid]).astype(np.int)

        features = dict(
            pos=torch.from_numpy(coord_data), z=torch.from_numpy(embed_data)
        )

        if self.has_energies:
            energy_data = np.array(
                np.load(self.energyfiles[fileid], mmap_mode="r")[index]
            )
            features["y"] = torch.from_numpy(energy_data)

        if self.has_forces:
            force_data = np.array(
                np.load(self.forcefiles[fileid], mmap_mode="r")[index]
            )
            features["dy"] = torch.from_numpy(force_data)

        return Data(**features)

    def len(self):
        return len(self.index)
