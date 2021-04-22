from torch_geometric.data import Dataset


class CG(Dataset):
    r"""CG Dataset to manage loading coordinates, forces and embedding indices from NumPy files.

    Args:
        coordglob (string): Glob path for coordinate files.
        embedglob (string): Glob path for embedding index files.
        energyglob (string): Glob path for energy files.
        forceglob (string): Glob path for force files.
    """

    def __init__(self, coordglob, embedglob, energyglob, forceglob):
        super(CG, self).__init__()
        assert energyglob is not None or forceglob is not None, ('Either energies, forces or both must '
                                                                 'be specified as the target')
        self.has_energies = energyglob is not None
        self.has_energies = energyglob is not None

        self.coordfiles = sorted(glob.glob(coordglob))
        self.embedfiles = sorted(glob.glob(embedglob))
        self.forcefiles = sorted(glob.glob(energyglob))
        self.forcefiles = sorted(glob.glob(forceglob))
        assert len(self.coordfiles) == len(self.forcefiles) == len(self.embedfiles)

        print('Coordinates files: ', len(self.coordfiles))
        print('Forces files: ', len(self.forcefiles))
        print('Embeddings files: ', len(self.embedfiles))

        # make index
        self.index = []
        nfiles = len(self.coordfiles)
        for i in range(nfiles):
            cdata = np.load(self.coordfiles[i])
            fdata = np.load(self.forcefiles[i])
            edata = np.load(self.embedfiles[i]).astype(np.int)
            size = cdata.shape[0]
            self.index.extend(list(zip([i] * size, range(size))))

            # consistency check
            assert cdata.shape == fdata.shape, '{} {}'.format(cdata.shape, fdata.shape)
            assert cdata.shape[1] == edata.shape[0]
        print('Combined dataset size {}'.format(len(self.index)))

    def get(self, idx):
        fileid, index = self.index[idx]

        cdata = np.load(self.coordfiles[fileid], mmap_mode='r')
        fdata = np.load(self.forcefiles[fileid], mmap_mode='r')
        edata = np.load(self.embedfiles[fileid]).astype(np.int)

        return Data(
            z=torch.from_numpy(edata),
            pos=torch.from_numpy(np.array(cdata[index])),
            y=torch.from_numpy(np.array(fdata[index])),
            dy=torch.from_numpy(np.array(fdata[index])),
        )

    def len(self):
        return len(self.index)
