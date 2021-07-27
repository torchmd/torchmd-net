from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchmdnet import datasets
from torchmdnet.utils import make_splits


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self.hparams = hparams
        self._mean = None
        self._std = None
        self._saved_dataloaders = dict()

    def setup(self, stage):
        if self.hparams.dataset == "Custom":
            self.dataset = datasets.Custom(
                self.hparams.coord_files,
                self.hparams.embed_files,
                self.hparams.energy_files,
                self.hparams.force_files,
            )
        else:
            self.dataset = getattr(datasets, self.hparams.dataset)(
                self.hparams.dataset_root, dataset_arg=self.hparams.dataset_arg
            )

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            self.hparams.train_size,
            self.hparams.val_size,
            self.hparams.test_size,
            self.hparams.seed,
            join(self.hparams.log_dir, "splits.npz"),
            self.hparams.splits,
        )
        print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_val)
        self.test_dataset = Subset(self.dataset, idx_test)

        if self.hparams.standardize:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.hparams.test_interval == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
            store_dataloader and not self.trainer.reload_dataloaders_every_epoch
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams.inference_batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        ys = torch.cat([batch.y.clone() for batch in data])

        self._mean = ys.mean()
        self._std = ys.std()


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
