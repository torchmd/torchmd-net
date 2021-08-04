import pytest
from pytest import mark
import torch
from torchmdnet import models
from torchmdnet.models.model import create_model
from torchmdnet.models import output_modules

from utils import load_example_args, create_example_batch


@mark.parametrize("model_name", models.__all__)
def test_forward(model_name):
    z, pos, batch = create_example_batch()
    model = create_model(load_example_args(model_name, prior=False))
    model(z, pos, batch=batch)
    model(z, pos, batch=None)


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("output_module_name", output_modules.__all__)
def test_forward_output_modules(model_name, output_module_name):
    z, pos, batch = create_example_batch()
    args = load_example_args(model_name, prior=False)
    args["output_module"] = output_module_name
    model = create_model(args)
    model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
def test_forward_torchscript(model_name):
    if model_name == "equivariant-transformer":
        # TODO: also test equivariant Transformer after the release of
        # https://github.com/rusty1s/pytorch_geometric/commit/673f94729b6a520b994699da5aa8dd3d1a1f670b
        pytest.skip("currently only works on the torch-geometric main branch")

    z, pos, batch = create_example_batch()
    model = torch.jit.script(create_model(load_example_args(model_name, prior=False)))
    model(z, pos, batch=batch)
