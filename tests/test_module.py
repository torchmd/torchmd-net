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
def test_train(model_name, use_atomref, precision, tmpdir):
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
    )
    datamodule = DataModule(args, DummyDataset(has_atomref=use_atomref))

    prior = None
    if use_atomref:
        prior = getattr(priors, args["prior_model"])(dataset=datamodule.dataset)
        args["prior_args"] = prior.get_init_args()

    module = LNNP(args, prior_model=prior)

    trainer = pl.Trainer(max_steps=10, default_root_dir=tmpdir, precision=args["precision"],inference_mode=False)
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)
