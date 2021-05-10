from torch.utils.data import Dataset


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
