import pytest
from pytest import mark
import pickle
from os.path import exists, dirname, join
import torch
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model
from torchmdnet.models import output_modules

from utils import load_example_args, create_example_batch


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("use_batch", [True, False])
@mark.parametrize("explicit_q_s", [True, False])
@mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward(model_name, use_batch, explicit_q_s, dtype):
    z, pos, batch = create_example_batch()
    pos = pos.to(dtype=dtype)
    model = create_model(load_example_args(model_name, prior_model=None, dtype=dtype))
    batch = batch if use_batch else None
    if explicit_q_s:
        model(z, pos, batch=batch, q=None, s=None)
    else:
        model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("output_model", output_modules.__all__)
@mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_output_modules(model_name, output_model, dtype):
    z, pos, batch = create_example_batch()
    args = load_example_args(model_name, remove_prior=True, output_model=output_model, dtype=dtype)
    model = create_model(args)
    model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all__)
@mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_torchscript(model_name, dtype):
    if model_name == "tensornet":
        pytest.skip("TensorNet does not support torchscript.")
    z, pos, batch = create_example_batch()
    model = torch.jit.script(
        create_model(load_example_args(model_name, remove_prior=True, derivative=True, dtype=dtype))
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
    pl.seed_everything(1234)

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
    if model_name not in expected or output_model not in expected[model_name]:
        raise ValueError(
            "Model not found in reference outputs, consider running this test with overwrite_reference=True."
        )

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
def test_gradients(model_name):
    pl.seed_everything(1234)
    dtype = torch.float64
    output_model = "Scalar"
    # create model and sample batch
    derivative = output_model in ["Scalar", "EquivariantScalar"]
    args = load_example_args(
        model_name,
        remove_prior=True,
        output_model=output_model,
        derivative=derivative,
        dtype=dtype,
    )
    model = create_model(args)
    z, pos, batch = create_example_batch(n_atoms=5)
    pos.requires_grad_(True)
    pos = pos.to(dtype)
    torch.autograd.gradcheck(
        model, (z, pos, batch), eps=1e-4, atol=1e-3, rtol=1e-2, nondet_tol=1e-3
    )
