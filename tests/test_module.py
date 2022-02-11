from pytest import mark
from glob import glob
from os.path import dirname, join
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import load_model
from torchmdnet.priors import Atomref
from torchmdnet.module import LNNP
from torchmdnet.data import DataModule

from utils import load_example_args, DummyDataset


@mark.parametrize("model_name", models.__all__)
def test_create_model(model_name):
    LNNP(load_example_args(model_name), prior_model=Atomref(100))


def test_load_model():
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
    load_model(checkpoint)


@mark.parametrize("model_name", models.__all__)
def test_train(model_name, tmpdir):
    args = load_example_args(
        model_name,
        remove_prior=True,
        train_size=0.8,
        val_size=0.05,
        test_size=None,
        log_dir=tmpdir,
        derivative=True,
        embedding_dimension=32,
        num_layers=3,
        num_rbf=16,
    )
    module = LNNP(args)
    datamodule = DataModule(args, DummyDataset())
    trainer = pl.Trainer(max_steps=10, default_root_dir=tmpdir)
    trainer.fit(module, datamodule)
    trainer.test()
