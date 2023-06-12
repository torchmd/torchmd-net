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



# Creates a skew-symmetric tensor from a vector
def vector_to_skewtensor_naive(vector):
    tensor = torch.cross(
        *torch.broadcast_tensors(
            vector[..., None], torch.eye(3, 3, device=vector.device)[None, None]
        )
    )
    return tensor.squeeze(0)

def vector_to_skewtensor(vector):
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device)
    tensor = torch.stack((zero, -vector[:,2], vector[:,1], vector[:,2], zero, -vector[:,0], -vector[:,1], vector[:,0], zero), dim=1)
    tensor = tensor.view(-1,3,3)
    return tensor.squeeze(0)

def test_skewtensor():
    vec = torch.rand(4, 3)
    skew = vector_to_skewtensor(vec)
    skew_naive = vector_to_skewtensor_naive(vec)
    print("Vector")
    print(vec)
    print("Naive")
    print(skew_naive)
    print("Optimized")
    print(skew)
    assert torch.allclose(skew, skew_naive)

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
