# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch.testing import assert_allclose
import pytest
from os.path import dirname, join
from torchmdnet.calculators import External
from torchmdnet.models.model import load_model, create_model

from utils import create_example_batch


@pytest.mark.parametrize("box", [None, torch.eye(3)])
@pytest.mark.parametrize("use_cuda_graphs", [True, False])
def test_compare_forward(box, use_cuda_graphs):
    if use_cuda_graphs and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
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
    model = create_model(args).to(device=device)
    z, pos, _ = create_example_batch(multiple_batches=False)
    z = z.to(device)
    pos = pos.to(device)
    calc = External(checkpoint, z.unsqueeze(0), use_cuda_graph=False, device=device)
    calc_graph = External(
        checkpoint, z.unsqueeze(0), use_cuda_graph=use_cuda_graphs, device=device
    )
    calc.model = model
    calc_graph.model = model
    if box is not None:
        box = (box * 2 * args["cutoff_upper"]).unsqueeze(0)
    for _ in range(10):
        e_calc, f_calc = calc.calculate(pos, box)
        e_pred, f_pred = calc_graph.calculate(pos, box)
        assert_allclose(e_calc, e_pred)
        assert_allclose(f_calc, f_pred)


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

    assert_allclose(e_calc, e_pred)
    assert_allclose(f_calc, f_pred.view(-1, len(z1), 3))
