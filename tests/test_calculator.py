# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch.testing import assert_close
import pytest
from os.path import dirname, join
from torchmdnet.calculators import External
from torchmdnet.models.model import load_model, create_model

from utils import create_example_batch

# Set relative and absolute tolerance values for float32 precision
# The original test used assert_allclose, which is now deprecated.
# assert_close is used instead, with default tolerances of 1e-5 (rtol) and 1.3e-6 (atol) for torch.float32.
# Here, we manually set rtol and atol to match the original test's tolerances.
rtol = 1e-4
atol = 1e-5

@pytest.mark.parametrize("box", [None, torch.eye(3)])
@pytest.mark.parametrize("use_cuda_graphs", [True, False])
def test_compare_forward(box, use_cuda_graphs):
    if use_cuda_graphs and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    checkpoint = join(dirname(dirname(__file__)), "tests", "tn_example.ckpt")
    args = {
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
        "output_model": "Scalar",
        "reduce_op": "sum",
        "precision": 32,
    }
    device = "cpu" if not use_cuda_graphs else "cuda"
    c_model = load_model(checkpoint).to(device=device)
    g_model = load_model(checkpoint, check_errors=not use_cuda_graphs, static_shapes=use_cuda_graphs).to(device=device)
    z, pos, _ = create_example_batch(multiple_batches=False)
    z = z.to(device)
    pos = pos.to(device)
    calc = External(c_model, z.unsqueeze(0), use_cuda_graph=False, device=device)
    calc_graph = External(
        g_model, z.unsqueeze(0), use_cuda_graph=use_cuda_graphs, device=device
    )
    
    if box is not None:
        box = (box * 2 * args["cutoff_upper"]).unsqueeze(0)
    
    for _ in range(10):
        e_calc, f_calc = calc.calculate(pos, box)
        e_pred, f_pred = calc_graph.calculate(pos, box)
        assert_close(e_calc, e_pred, rtol=rtol, atol=atol)
        assert_close(f_calc, f_pred, rtol=rtol, atol=atol)

def test_compare_forward_multiple():
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
    z1, pos1, _ = create_example_batch(multiple_batches=False)
    z2, pos2, _ = create_example_batch(multiple_batches=False)
    calc = External(checkpoint, torch.stack([z1, z2], dim=0))
    model = load_model(checkpoint, derivative=True)

    e_calc, f_calc = calc.calculate(torch.cat([pos1, pos2], dim=0), None)
    e_pred, f_pred = model(
        torch.cat([z1, z2]),
        torch.cat([pos1, pos2], dim=0),
        torch.cat([torch.zeros(len(z1)), torch.ones(len(z2))]).long(),
    )

    assert_close(e_calc, e_pred, rtol=rtol, atol=atol)
    assert_close(f_calc, f_pred.view(-1, len(z1), 3), rtol=rtol, atol=atol)
