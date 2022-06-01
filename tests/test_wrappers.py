import pytest
from pytest import mark
from torchmdnet import models
from torchmdnet.models.model import create_model
from torchmdnet.models.wrappers import AtomFilter
from utils import load_example_args, create_example_batch


@mark.parametrize("remove_threshold", [-1, 2, 5])
@mark.parametrize("model_name", models.__all__)
def test_atom_filter(remove_threshold, model_name):
    # wrap a representation model using the AtomFilter wrapper
    model = create_model(load_example_args(model_name, remove_prior=True))
    model = model.representation_model
    model = AtomFilter(model, remove_threshold)

    z, pos, batch = create_example_batch(n_atoms=100)
    x, v, z, pos, batch = model(z, pos, batch, None, None)

    assert (z > remove_threshold).all(), (
        f"Lowest updated atomic number is {z.min()} but "
        f"the atom filter is set to {remove_threshold}"
    )
    assert len(z) == len(
        pos
    ), "Number of z and pos values doesn't match after AtomFilter"
    assert len(z) == len(
        batch
    ), "Number of z and batch values doesn't match after AtomFilter"
