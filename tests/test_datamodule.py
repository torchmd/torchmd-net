from pytest import mark
import torch
from torchmdnet.data import DataModule
from utils import load_example_args, DummyDataset


def test_datamodule_create(tmpdir):
    args = load_example_args("graph-network")
    args["train_size"] = 800
    args["val_size"] = 100
    args["test_size"] = 100
    args["log_dir"] = tmpdir

    dataset = DummyDataset()
    data = DataModule(args, dataset=dataset)
    data.prepare_data()
    data.setup("fit")

    data._get_dataloader(data.train_dataset, "train", store_dataloader=False)
    data._get_dataloader(data.val_dataset, "val", store_dataloader=False)
    data._get_dataloader(data.test_dataset, "test", store_dataloader=False)

    dl1 = data._get_dataloader(data.train_dataset, "train", store_dataloader=False)
    dl2 = data._get_dataloader(data.train_dataset, "train", store_dataloader=False)
    assert dl1 is not dl2


@mark.parametrize("energy,forces", [(True, True), (True, False), (False, True)])
@mark.parametrize("has_atomref", [True, False])
def test_datamodule_standardize(energy, forces, has_atomref, tmpdir):
    args = load_example_args("graph-network")
    args["standardize"] = True
    args["train_size"] = 800
    args["val_size"] = 100
    args["test_size"] = 100
    args["log_dir"] = tmpdir

    dataset = DummyDataset(energy=energy, forces=forces, has_atomref=has_atomref)
    data = DataModule(args, dataset=dataset)
    data.prepare_data()
    data.setup("fit")

    assert (data.atomref is not None) == has_atomref
    if has_atomref:
        assert (data.atomref == dataset.get_atomref()).all()

    if energy:
        train_energies = torch.tensor(dataset.energies)[data.idx_train]
        if has_atomref:
            # the mean and std should be computed after removing atomrefs
            train_energies -= torch.tensor(
                [
                    dataset.atomref[zs].sum()
                    for i, zs in enumerate(dataset.z)
                    if i in data.idx_train
                ]
            )
        # mean and std attributes should provide mean and std of the training split
        assert torch.allclose(data.mean, train_energies.mean())
        # assert torch.allclose(data.std, train_energies.std())
    else:
        # the data module should not have mean and std set if the dataset does not include energies
        assert data.mean is None and data.std is None
