# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from pytest import mark
from glob import glob
from os.path import dirname, join
import lightning as pl
from torchmdnet import models
from torchmdnet.models.model import load_model
from torchmdnet.priors import Atomref
from torchmdnet.module import LNNP
from torchmdnet.data import DataModule
from torchmdnet import priors
import os

from utils import load_example_args, DummyDataset


@mark.parametrize("model_name", models.__all_models__)
def test_create_model(model_name):
    LNNP(load_example_args(model_name), prior_model=Atomref(100))


def test_load_model():
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
    load_model(checkpoint)


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("use_atomref", [True, False])
@mark.parametrize("precision", [32, 64])
@mark.skipif(
    os.getenv("LONG_TRAIN", "false") == "false", reason="Skipping long train test"
)
def test_train(model_name, use_atomref, precision, tmpdir):
    import torch

    torch.set_num_threads(1)

    accelerator = "auto"
    if os.getenv("CPU_TRAIN", "false") == "true":
        # OSX MPS backend runs out of memory on Github Actions
        torch.set_default_device("cpu")
        accelerator = "cpu"

    args = load_example_args(
        model_name,
        remove_prior=not use_atomref,
        train_size=0.8,
        val_size=0.05,
        test_size=None,
        log_dir=tmpdir,
        derivative=True,
        embedding_dimension=16,
        num_layers=2,
        num_rbf=16,
        batch_size=8,
        precision=precision,
        num_workers=0,
    )
    datamodule = DataModule(args, DummyDataset(has_atomref=use_atomref))

    prior = None
    if use_atomref:
        prior = getattr(priors, args["prior_model"])(dataset=datamodule.dataset)
        args["prior_args"] = prior.get_init_args()

    module = LNNP(args, prior_model=prior)

    trainer = pl.Trainer(
        max_steps=10,
        default_root_dir=tmpdir,
        precision=args["precision"],
        inference_mode=False,
        accelerator=accelerator,
        num_nodes=1,
        devices=1,
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("use_atomref", [True, False])
@mark.parametrize("precision", [32, 64])
def test_dummy_train(model_name, use_atomref, precision, tmpdir):
    import torch

    torch.set_num_threads(1)

    accelerator = "auto"
    if os.getenv("CPU_TRAIN", "false") == "true":
        # OSX MPS backend runs out of memory on Github Actions
        torch.set_default_device("cpu")
        accelerator = "cpu"

    extra_args = {}
    if model_name != "tensornet":
        extra_args["num_heads"] = 2

    args = load_example_args(
        model_name,
        remove_prior=not use_atomref,
        train_size=0.05,
        val_size=0.01,
        test_size=0.01,
        log_dir=tmpdir,
        derivative=True,
        embedding_dimension=2,
        num_layers=1,
        num_rbf=4,
        batch_size=2,
        precision=precision,
        num_workers=0,
        **extra_args,
    )
    datamodule = DataModule(args, DummyDataset(has_atomref=use_atomref))

    prior = None
    if use_atomref:
        prior = getattr(priors, args["prior_model"])(dataset=datamodule.dataset)
        args["prior_args"] = prior.get_init_args()

    module = LNNP(args, prior_model=prior)

    trainer = pl.Trainer(
        max_steps=10,
        default_root_dir=tmpdir,
        precision=args["precision"],
        inference_mode=False,
        accelerator=accelerator,
        num_nodes=1,
        devices=1,
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)
