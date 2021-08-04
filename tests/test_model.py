from pytest import mark
from glob import glob
from os.path import dirname, join
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model, load_model
from torchmdnet.priors import Atomref
from torchmdnet.module import LNNP

from utils import load_example_args


@mark.parametrize("model_name", models.__all__)
def test_create_model(model_name):
    LNNP(load_example_args(model_name), prior_model=Atomref(100))


@mark.parametrize(
    "checkpoint",
    glob(
        join(dirname(dirname(__file__)), "examples", "pretrained", "**", "*.ckpt"),
        recursive=True,
    ),
)
def test_load_model(checkpoint):
    load_model(checkpoint)


@mark.parametrize("model_name", models.__all__)
def test_seed(model_name):
    pl.seed_everything(1234)
    m1 = create_model(load_example_args(model_name), prior_model=Atomref(100))
    pl.seed_everything(1234)
    m2 = create_model(load_example_args(model_name), prior_model=Atomref(100))

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert (p1 == p2).all(), "Parameters don't match although using the same seed."
