import glob
import numpy as np

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn.models.schnet import qm9_target_dict


class CGDataset(Dataset):
    r"""CG Dataset to manage loading coordinates, forces and embedding indices from NumPy files.

    Args:
        coordglob (string): Glob path for coordinate files.
        forceglob (string): Glob path for force files.
        embedglob (string): Glob path for embedding index files.
    """

    def __init__(self, coordglob, forceglob, embedglob):
        self.coordfiles = sorted(glob.glob(coordglob))
        self.forcefiles = sorted(glob.glob(forceglob))
        self.embedfiles = sorted(glob.glob(embedglob))
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

    def __getitem__(self, idx):
        fileid, index = self.index[idx]

        cdata = np.load(self.coordfiles[fileid], mmap_mode='r')
        fdata = np.load(self.forcefiles[fileid], mmap_mode='r')
        edata = np.load(self.embedfiles[fileid]).astype(np.int)

        return Data(
            pos=torch.from_numpy(np.array(cdata[index])),
            y=torch.from_numpy(np.array(fdata[index])),
            z=torch.from_numpy(edata)
        )

    def __len__(self):
        return len(self.index)


class Subset(Dataset):
    r"""Subset of a bigger dataset, given a list of indices.

    Arguments:
        dataset (Dataset): The complete dataset
        indices (array-like): Sequence of indices defining the subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]

    def __len__(self):
        return len(self.indices)


class AtomrefDataset(Dataset):
    r"""Dataset wrapper which removes the atomrefs from labels.

    Arguments:
        dataset (Dataset): A dataset with property `z` to index the atomrefs.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.atomref = self.dataset.get_atomref()

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item.y -= self.atomref[item.z].sum()
        return item

    def __len__(self):
        return len(self.dataset)
