import pytest
from pytest import mark
import pickle
from os.path import exists, dirname, join
import torch
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models import output_modules

from utils import load_example_args, create_example_batch


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("use_batch", [True, False])
@mark.parametrize("explicit_q_s", [True, False])
def test_forward(model_name, use_batch, explicit_q_s):
    z, pos, batch = create_example_batch()
    model = create_model(load_example_args(model_name, prior_model=None))
    batch = batch if use_batch else None
    if explicit_q_s:
        model(z, pos, batch=batch, q=None, s=None)
    else:
        model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("output_model", output_modules.__all__)
def test_forward_output_modules(model_name, output_model):
    z, pos, batch = create_example_batch()
    args = load_example_args(model_name, remove_prior=True, output_model=output_model)
    model = create_model(args)
    model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
def test_forward_torchscript(model_name):
    z, pos, batch = create_example_batch()
    model = torch.jit.script(
        create_model(load_example_args(model_name, remove_prior=True, derivative=True))
    )
    model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
def test_seed(model_name):
    args = load_example_args(model_name, remove_prior=True)
    pl.seed_everything(1234)
    m1 = create_model(args)
    pl.seed_everything(1234)
    m2 = create_model(args)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert (p1 == p2).all(), "Parameters don't match although using the same seed."


@mark.parametrize("model_name", models.__all__)
@mark.parametrize(
    "output_model",
    output_modules.__all__,
)
def test_forward_output(model_name, output_model, overwrite_reference=False):
    pl.seed_everynthing(1234)

    # create model and sample batch
    derivative = output_model in ["Scalar", "EquivariantScalar"]
    args = load_example_args(
        model_name,
        remove_prior=True,
        output_model=output_model,
        derivative=derivative,
    )
    model = create_model(args)
    z, pos, batch = create_example_batch(n_atoms=5)

    # run step
    pred, deriv = model(z, pos, batch)

    # load reference outputs
    expected_path = join(dirname(__file__), "expected.pkl")
    assert exists(expected_path), "Couldn't locate reference outputs."
    with open(expected_path, "rb") as f:
        expected = pickle.load(f)

    if overwrite_reference:
        # this overwrites the previous reference outputs and shouldn't be executed during testing
        if model_name in expected:
            expected[model_name][output_model] = dict(pred=pred, deriv=deriv)
        else:
            expected[model_name] = {output_model: dict(pred=pred, deriv=deriv)}

        with open(expected_path, "wb") as f:
            pickle.dump(expected, f)
        assert (
            False
        ), f"Set new reference outputs for {model_name} with output model {output_model}."

    # compare actual ouput with reference
    torch.testing.assert_allclose(pred, expected[model_name][output_model]["pred"])
    if derivative:
        torch.testing.assert_allclose(
            deriv, expected[model_name][output_model]["deriv"]
        )

@mark.parametrize("model_name", models.__all__)
@mark.parametrize(
    "output_model",
    output_modules.__all__,
)
def test_gradients(model_name, output_model):
    pl.seed_everything(1234)
    if model_name != "equivariant-transformer":
        pytest.skip("Gradients are not implemented for this model.")
    # create model and sample batch
    derivative = output_model in ["Scalar", "EquivariantScalar"]
    args = load_example_args(
        model_name,
        remove_prior=True,
        output_model=output_model,
        derivative=derivative,
    )
    model = create_model(args).to(torch.float64)
    print(model)
    z, pos, batch = create_example_batch(n_atoms=5)
    pos.requires_grad = True
    pos.to(torch.float64)
    torch.autograd.gradcheck(model, (z, pos, batch), eps=1e-4, atol=1e-3, rtol=1e-2, nondet_tol=1e-3)
