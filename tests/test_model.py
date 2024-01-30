# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pytest
from pytest import mark
import pickle
from os.path import exists, dirname, join
import torch
import lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model
from torchmdnet.models import output_modules
from torchmdnet.models.utils import dtype_mapping

from utils import load_example_args, create_example_batch


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("use_batch", [True, False])
@mark.parametrize("explicit_q_s", [True, False])
@mark.parametrize("precision", [32, 64])
def test_forward(model_name, use_batch, explicit_q_s, precision):
    z, pos, batch = create_example_batch()
    pos = pos.to(dtype=dtype_mapping[precision])
    model = create_model(load_example_args(model_name, prior_model=None, precision=precision))
    batch = batch if use_batch else None
    if explicit_q_s:
        model(z, pos, batch=batch, q=None, s=None)
    else:
        model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("output_model", output_modules.__all__)
@mark.parametrize("precision", [32,64])
def test_forward_output_modules(model_name, output_model, precision):
    z, pos, batch = create_example_batch()
    args = load_example_args(model_name, remove_prior=True, output_model=output_model, precision=precision)
    model = create_model(args)
    model(z, pos, batch=batch)


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("device", ["cpu", "cuda"])
def test_torchscript(model_name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    z, pos, batch = create_example_batch()
    z = z.to(device)
    pos = pos.to(device)
    batch = batch.to(device)
    model = torch.jit.script(
        create_model(load_example_args(model_name, remove_prior=True, derivative=True))
    ).to(device=device)
    y, neg_dy = model(z, pos, batch=batch)
    grad_outputs = [torch.ones_like(neg_dy)]
    ddy = torch.autograd.grad(
        [neg_dy],
        [pos],
        grad_outputs=grad_outputs,
    )[0]

@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("device", ["cpu", "cuda"])
def test_torchscript_dynamic_shapes(model_name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if model_name == "tensornet":
        pytest.skip("TorchScripted TensorNet does not support dynamic shapes.")
    z, pos, batch = create_example_batch()
    model = torch.jit.script(
        create_model(load_example_args(model_name, remove_prior=True, derivative=True))
    ).to(device=device)
    #Repeat the input to make it dynamic
    for rep in range(0, 5):
        print(rep)
        zi = z.repeat_interleave(rep+1, dim=0).to(device=device)
        posi = pos.repeat_interleave(rep+1, dim=0).to(device=device)
        batchi = torch.randint(0, 10, (zi.shape[0],)).sort()[0].to(device=device)
        y, neg_dy = model(zi, posi, batch=batchi)
        grad_outputs = [torch.ones_like(neg_dy)]
        ddy = torch.autograd.grad(
            [neg_dy],
            [posi],
            grad_outputs=grad_outputs,
        )[0]

#Currently only tensornet is CUDA graph compatible
@mark.parametrize("model_name", ["tensornet"])
def test_cuda_graph_compatible(model_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    z, pos, batch = create_example_batch()
    args = {"model": model_name,
            "embedding_dimension": 128,
            "num_layers": 2,
            "num_rbf": 32,
            "rbf_type": "expnorm",
            "trainable_rbf": False,
            "activation": "silu",
            "cutoff_lower": 0.0,
            "cutoff_upper": 5.0,
            "max_z": 100,
            "max_num_neighbors": 128,
            "equivariance_invariance_group": "O(3)",
            "prior_model": None,
            "atom_filter": -1,
            "derivative": True,
            "check_error": False,
            "static_shapes": True,
            "output_model": "Scalar",
            "reduce_op": "sum",
            "precision": 32 }
    model = create_model(args).to(device="cuda")
    model.eval()
    z = z.to("cuda")
    pos = pos.to("cuda").requires_grad_(True)
    batch = batch.to("cuda")
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(0, 15):
            y, neg_dy = model(z, pos, batch=batch)
    g = torch.cuda.CUDAGraph()
    y2, neg_dy2 = model(z, pos, batch=batch)
    with torch.cuda.graph(g):
        y, neg_dy = model(z, pos, batch=batch)
    y.fill_(0.0)
    neg_dy.fill_(0.0)
    g.replay()
    assert torch.allclose(y, y2)
    assert torch.allclose(neg_dy, neg_dy2, atol=1e-5, rtol=1e-5)

@mark.parametrize("model_name", models.__all_models__)
def test_seed(model_name):
    args = load_example_args(model_name, remove_prior=True)
    pl.seed_everything(1234)
    m1 = create_model(args)
    pl.seed_everything(1234)
    m2 = create_model(args)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert (p1 == p2).all(), "Parameters don't match although using the same seed."

@mark.parametrize("model_name", models.__all_models__)
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
    torch.testing.assert_close(pred, expected[model_name][output_model]["pred"], atol=1e-5, rtol=1e-5)
    if derivative:
        torch.testing.assert_close(
            deriv, expected[model_name][output_model]["deriv"], atol=1e-5, rtol=1e-5
        )


@mark.parametrize("model_name", models.__all_models__)
def test_gradients(model_name):
    pl.seed_everything(12345)
    precision = 64
    output_model = "Scalar"
    # create model and sample batch
    derivative = output_model in ["Scalar", "EquivariantScalar"]
    args = load_example_args(
        model_name,
        remove_prior=True,
        output_model=output_model,
        derivative=derivative,
        precision=precision
    )
    model = create_model(args)
    z, pos, batch = create_example_batch(n_atoms=5)
    pos.requires_grad_(True)
    pos = pos.to(torch.float64)
    torch.autograd.gradcheck(
        model, (z, pos, batch), eps=1e-4, atol=1e-3, rtol=1e-2, nondet_tol=1e-3
    )
