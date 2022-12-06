import pytest
from pytest import mark
import torch
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model, create_prior_models
from torchmdnet.module import LNNP
from torchmdnet.priors import Atomref, D2, ZBL
from torch_scatter import scatter
from utils import load_example_args, create_example_batch, DummyDataset
from os.path import dirname, join
import tempfile


@mark.parametrize("model_name", models.__all__)
def test_atomref(model_name):
    dataset = DummyDataset(has_atomref=True)
    atomref = Atomref(max_z=100, dataset=dataset)
    z, pos, batch = create_example_batch()

    # create model with atomref
    pl.seed_everything(1234)
    model_atomref = create_model(
        load_example_args(model_name, prior_model="Atomref"), prior_model=atomref
    )
    # create model without atomref
    pl.seed_everything(1234)
    model_no_atomref = create_model(load_example_args(model_name, remove_prior=True))

    # get output from both models
    x_atomref, _ = model_atomref(z, pos, batch)
    x_no_atomref, _ = model_no_atomref(z, pos, batch)

    # check if the output of both models differs by the expected atomref contribution
    expected_offset = scatter(dataset.get_atomref().squeeze()[z], batch).unsqueeze(1)
    torch.testing.assert_allclose(x_atomref, x_no_atomref + expected_offset)

def test_zbl():
    pos = torch.tensor([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32)  # Atom positions in Bohr
    types = torch.tensor([0, 1, 2, 1], dtype=torch.long)  # Atom types
    atomic_number = torch.tensor([1, 6, 8], dtype=torch.int8)  # Mapping of atom types to atomic numbers
    distance_scale = 5.29177210903e-11  # Convert Bohr to meters
    energy_scale = 1000.0/6.02214076e23  # Convert kJ/mol to Joules

    # Use the ZBL class to compute the energy.

    zbl = ZBL(10.0, 5, atomic_number, distance_scale=distance_scale, energy_scale=energy_scale)
    energy = zbl.post_reduce(torch.zeros((1,)), types, pos, torch.zeros_like(types), {})[0]

    # Compare to the expected value.

    def compute_interaction(pos1, pos2, z1, z2):
        delta = pos1-pos2
        r = torch.sqrt(torch.dot(delta, delta))
        x = r / (0.8854/(z1**0.23 + z2**0.23))
        phi = 0.1818*torch.exp(-3.2*x) + 0.5099*torch.exp(-0.9423*x) + 0.2802*torch.exp(-0.4029*x) + 0.02817*torch.exp(-0.2016*x)
        cutoff = 0.5*(torch.cos(r*torch.pi/10.0) + 1.0)
        return cutoff*phi*(138.935/5.29177210903e-2)*z1*z2/r

    expected = 0
    for i in range(len(pos)):
        for j in range(i):
            expected += compute_interaction(pos[i], pos[j], atomic_number[types[i]], atomic_number[types[j]])
    torch.testing.assert_allclose(expected, energy)

def test_multiple_priors():
    # Create a model from a config file.

    dataset = DummyDataset(has_atomref=True)
    config_file = join(dirname(__file__), 'priors.yaml')
    args = load_example_args('equivariant-transformer', config_file=config_file)
    prior_models = create_prior_models(args, dataset)
    args['prior_args'] = [p.get_init_args() for p in prior_models]
    model = LNNP(args, prior_model=prior_models)
    priors = model.model.prior_model

    # Make sure the priors were created correctly.

    assert len(priors) == 3
    assert isinstance(priors[0], ZBL)
    assert isinstance(priors[1], D2)
    assert isinstance(priors[2], Atomref)
    assert priors[0].cutoff_distance == 4.0
    assert priors[0].max_num_neighbors == 50
    assert priors[1].cutoff_distance == 10.0
    assert priors[1].max_num_neighbors == 100

    # Save and load a checkpoint, and make sure the priors are correct.

    with tempfile.NamedTemporaryFile() as f:
        torch.save(model, f)
        f.seek(0)
        model2 = torch.load(f)
        priors2 = model2.model.prior_model
        assert len(priors2) == 3
        assert isinstance(priors2[0], ZBL)
        assert isinstance(priors2[1], D2)
        assert isinstance(priors2[2], Atomref)
        assert priors2[0].cutoff_distance == priors[0].cutoff_distance
        assert priors2[0].max_num_neighbors == priors[0].max_num_neighbors
        assert priors2[1].cutoff_distance == priors[1].cutoff_distance
        assert priors2[1].max_num_neighbors == priors[1].max_num_neighbors
