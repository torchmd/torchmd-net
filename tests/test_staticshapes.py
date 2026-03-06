import pytest
import torch
import lightning as pl
from torchmdnet.models.model import create_model

from utils import create_example_batch


def _make_args_tensornet2(static_shapes):
    return {
        "model": "tensornet2",
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
        "static_shapes": static_shapes,
        "output_model": "ScalarPlusWeightedCoulomb",
        "reduce_op": "sum",
        "precision": 32,
        "q_dim": 16,
        "q_weights": [1.0, 1.0, 1.0],  # num_layers + 1 entries required
    }


def _make_args_tensornet(static_shapes):
    return {
        "model": "tensornet",
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
        "static_shapes": static_shapes,
        "output_model": "Scalar",
        "reduce_op": "sum",
        "precision": 32,
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "make_args",
    [_make_args_tensornet2, _make_args_tensornet],
    ids=["tensornet2-ScalarPlusWeightedCoulomb", "tensornet-Scalar"],
)
def test_staticshapes(make_args, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pl.seed_everything(1234)
    model_dynamic = create_model(make_args(static_shapes=False)).to(device)

    pl.seed_everything(1234)
    model_static = create_model(make_args(static_shapes=True)).to(device)

    z, pos, batch = create_example_batch(n_atoms=10)
    z = z.to(device)
    pos = pos.to(device).requires_grad_(True)
    batch = batch.to(device)

    model_dynamic.eval()
    model_static.eval()

    y_dynamic, neg_dy_dynamic = model_dynamic(z, pos, batch=batch)
    y_static, neg_dy_static = model_static(z, pos, batch=batch)

    torch.testing.assert_close(y_dynamic, y_static, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(neg_dy_dynamic, neg_dy_static, atol=1e-5, rtol=1e-5)
