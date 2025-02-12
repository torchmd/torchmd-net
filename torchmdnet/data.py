# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException
from torchmdnet.models.utils import scatter
import warnings


class DataModule(LightningDataModule):
    """A LightningDataModule for loading datasets from the torchmdnet.datasets module.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the
            dataset. See the documentation of the torchmdnet.datasets module
            for details.
        dataset (torch_geometric.data.Dataset): A dataset to use instead of
            loading a new one from the torchmdnet.datasets module.
    """

    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

    def setup(self, stage):
        if self.dataset is None:
            if self.hparams["dataset"] == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams["coord_files"],
                    self.hparams["embed_files"],
                    self.hparams["energy_files"],
                    self.hparams["force_files"],
                    self.hparams["dataset_preload_limit"],
                )
            else:
                dataset_arg = {}
                if self.hparams["dataset_arg"] is not None:
                    dataset_arg = self.hparams["dataset_arg"]
                if self.hparams["dataset"] == "HDF5":
                    dataset_arg["dataset_preload_limit"] = self.hparams[
                        "dataset_preload_limit"
                    ]
                self.dataset = getattr(datasets, self.hparams["dataset"])(
                    self.hparams["dataset_root"], **dataset_arg
                )

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            # Mark as deprecated
            warnings.warn(
                "The standardize option is deprecated and will be removed in the future. ",
                DeprecationWarning,
            )
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        # To allow to report the performance on the testing dataset during training
        # we send the trainer two dataloaders every few steps and modify the
        # validation step to understand the second dataloader as test data.
        if self._is_test_during_training_epoch():
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        """Returns the atomref of the dataset if it has one, otherwise None."""
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        """Returns the mean of the dataset if it has one, otherwise None."""
        return self._mean

    @property
    def std(self):
        """Returns the standard deviation of the dataset if it has one, otherwise None."""
        return self._std

    def _is_test_during_training_epoch(self):
        return (
            len(self.test_dataset) > 0
            and self.hparams["test_interval"] > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        )

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]

        shuffle = stage == "train"
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            shuffle=shuffle,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if "y" not in batch or batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
