import pytest
from pytest import mark
import torch as pt
from torchmdnet.models.model import create_model
from torchmdnet.optimize import optimize
from torchmdnet.models.utils import dtype_mapping

@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("num_atoms", [10, 100])
def test_gn(device, num_atoms):

    if not pt.cuda.is_available() and device == "cuda":
        pytest.skip("No GPU")

    device = pt.device(device)

    # Generate random inputs
    elements = pt.randint(1, 100, (num_atoms,)).to(device)
    positions = (10 * pt.rand((num_atoms, 3)) - 5).to(device)

    # Crate a non-optimized model
    #   SchNet: TorchMD_GN(rbf_type='gauss', trainable_rbf=False, activation='ssp', neighbor_embedding=False)
    args = {
        "embedding_dimension": 128,
        "num_layers": 6,
        "num_rbf": 50,
        "rbf_type": "gauss",
        "trainable_rbf": False,
        "activation": "ssp",
        "neighbor_embedding": False,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "max_z": 100,
        "max_num_neighbors": num_atoms,
        "model": "graph-network",
        "aggr": "add",
        "derivative": True,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    ref_model = create_model(args).to(device)

    # Execute the non-optimized model
    ref_energy, ref_gradient = ref_model(elements, positions)

    # Optimize the model
    model = optimize(ref_model).to(device)
    positions.to(dtype_mapping[args["precision"]])
    # Execute the optimize model
    energy, gradient = model(elements, positions)

    pt.testing.assert_close(ref_energy, energy, rtol=1e-5, atol=1e-5)
    pt.testing.assert_close(ref_gradient, gradient, rtol=1e-4, atol=1e-5)
