# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pytest
from pytest import mark
import torch
import lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model, create_prior_models
from torchmdnet.module import LNNP
from torchmdnet.priors import Atomref, LearnableAtomref, D2, ZBL, Coulomb
from torchmdnet.models.utils import scatter
from utils import load_example_args, create_example_batch, DummyDataset
from os.path import dirname, join
import tempfile


@mark.parametrize("model_name", models.__all_models__)
@mark.parametrize("enable_atomref", [True, False])
def test_atomref(model_name, enable_atomref):
    dataset = DummyDataset(has_atomref=True)
    atomref = Atomref(max_z=100, dataset=dataset, enable=enable_atomref)
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
    if enable_atomref:
        expected_offset = scatter(dataset.get_atomref().squeeze()[z], batch).unsqueeze(
            1
        )
    else:
        expected_offset = 0
    torch.testing.assert_close(x_atomref, x_no_atomref + expected_offset)


@mark.parametrize("trainable", [True, False])
def test_atomref_trainable(trainable):
    dataset = DummyDataset(has_atomref=True)
    atomref = Atomref(max_z=100, dataset=dataset, trainable=trainable)
    assert atomref.atomref.weight.requires_grad == trainable


def test_learnableatomref():
    atomref = LearnableAtomref(max_z=100)
    assert atomref.atomref.weight.requires_grad == True


def test_zbl():
    pos = torch.tensor(
        [[1.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )  # Atom positions in Bohr
    types = torch.tensor([0, 1, 2, 1], dtype=torch.long)  # Atom types
    atomic_number = torch.tensor(
        [1, 6, 8], dtype=torch.int8
    )  # Mapping of atom types to atomic numbers
    distance_scale = 5.29177210903e-11  # Convert Bohr to meters
    energy_scale = 1000.0 / 6.02214076e23  # Convert kJ/mol to Joules

    # Use the ZBL class to compute the energy.

    zbl = ZBL(
        10.0, 5, atomic_number, distance_scale=distance_scale, energy_scale=energy_scale
    )
    energy = zbl.post_reduce(
        torch.zeros((1,)), types, pos, torch.zeros_like(types), None, {}
    )[0]

    # Compare to the expected value.

    def compute_interaction(pos1, pos2, z1, z2):
        delta = pos1 - pos2
        r = torch.sqrt(torch.dot(delta, delta))
        x = r / (0.8854 / (z1**0.23 + z2**0.23))
        phi = (
            0.1818 * torch.exp(-3.2 * x)
            + 0.5099 * torch.exp(-0.9423 * x)
            + 0.2802 * torch.exp(-0.4029 * x)
            + 0.02817 * torch.exp(-0.2016 * x)
        )
        cutoff = 0.5 * (torch.cos(r * torch.pi / 10.0) + 1.0)
        return cutoff * phi * (138.935 / 5.29177210903e-2) * z1 * z2 / r

    expected = 0
    for i in range(len(pos)):
        for j in range(i):
            expected += compute_interaction(
                pos[i], pos[j], atomic_number[types[i]], atomic_number[types[j]]
            )
    torch.testing.assert_close(expected, energy, rtol=1e-4, atol=1e-4)


# Convert kJ/mol and eV to Joules.
@pytest.mark.parametrize("energy_scale", [1000.0 / 6.02214076e23, 1.602176634e-19])
@pytest.mark.parametrize(
    "distance_scale", [1e-10, 1e-9, 5.29177210903e-11]
)  # Convert Angstrom, nm, and Bohr to meters
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coulomb(distance_scale, energy_scale, dtype):
    lower_switch_distance = 0.9  # In nanometers
    upper_switch_distance = 1.3  # In nanometers
    # Physical system, with pair distances of 0.640, 0.854, 1.0, 1.063, 1.2, and 1.552 nm,
    # spanning below, within, and above the switch region, with attractive and repulsive pairs.
    pos_nm = torch.tensor(
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.8, 0.8, 0.0], [0.0, 0.0, -0.4]],
        dtype=torch.float64,
    )
    charge = torch.tensor([0.2, -0.1, 0.8, -0.9], dtype=torch.float64)  # In units of e
    types = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    # Express the same physical system in the dataset's distance units.
    pos = (pos_nm / (distance_scale * 1e9)).to(dtype)
    pos.requires_grad_(True)
    charge = charge.to(dtype)

    # Use the Coulomb class to compute the energy and forces through autograd.
    coulomb = Coulomb(
        lower_switch_distance,
        upper_switch_distance,
        5,
        distance_scale=distance_scale,
        energy_scale=energy_scale,
    )
    energy = coulomb.post_reduce(
        torch.zeros((1,)),
        types,
        pos,
        torch.zeros_like(types),
        extra_args={"partial_charges": charge},
    )[0]
    force = -torch.autograd.grad(energy, pos)[0]

    # Compare to analytic values computed in physical units (Joules and nanometers),
    # independent of the dataset's units.
    k = 1e9 * 2.30707e-28  # Coulomb constant times e^2, in J*nm
    nm_per_distance_unit = distance_scale * 1e9

    expected_energy = torch.zeros((), dtype=torch.float64)
    expected_force = torch.zeros_like(
        pos_nm
    )  # Force in dataset energy units per distance unit
    # Sum over ordered pairs: factor 0.5 for the energy because each pair appears twice, no
    # factor for the force because this sum is already the gradient of the total energy.
    for i in range(len(pos_nm)):
        for j in range(len(pos_nm)):
            if i == j:
                continue
            delta = pos_nm[i] - pos_nm[j]
            r = torch.sqrt(torch.dot(delta, delta))
            phase = (
                torch.clamp(r, min=lower_switch_distance, max=upper_switch_distance)
                - lower_switch_distance
            ) / (upper_switch_distance - lower_switch_distance)
            switch = 0.5 - 0.5 * torch.cos(torch.pi * phase)
            dswitch = (
                0.5
                * torch.pi
                * torch.sin(torch.pi * phase)
                / (upper_switch_distance - lower_switch_distance)
            )
            expected_energy = (
                expected_energy
                + 0.5 * k * charge[i] * charge[j] * switch / r / energy_scale
            )
            # Derivative of the physical pair energy with respect to the separation vector,
            # in J/nm.  Chain rule to dataset units: nm per dataset distance unit, then the
            # energy scale.
            denergy_ddelta = (
                k * charge[i] * charge[j] * (dswitch * r - switch) * delta / r**3
            )
            expected_force[i] += -nm_per_distance_unit * denergy_ddelta / energy_scale

    torch.testing.assert_close(
        expected_energy, energy, rtol=1e-4, atol=1e-4, check_dtype=False
    )
    torch.testing.assert_close(
        expected_force,
        force,
        rtol=1e-4,
        atol=1e-4 * float(expected_force.abs().max()),
        check_dtype=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multiple_priors(dtype):
    # Create a model from a config file.

    dataset = DummyDataset(has_atomref=True)
    config_file = join(dirname(__file__), "priors.yaml")
    args = load_example_args(
        "equivariant-transformer", config_file=config_file, dtype=dtype
    )
    prior_models = create_prior_models(args, dataset)
    args["prior_args"] = [p.get_init_args() for p in prior_models]
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
        model2 = torch.load(f, weights_only=False)
        priors2 = model2.model.prior_model
        assert len(priors2) == 3
        assert isinstance(priors2[0], ZBL)
        assert isinstance(priors2[1], D2)
        assert isinstance(priors2[2], Atomref)
        assert priors2[0].cutoff_distance == priors[0].cutoff_distance
        assert priors2[0].max_num_neighbors == priors[0].max_num_neighbors
        assert priors2[1].cutoff_distance == priors[1].cutoff_distance
        assert priors2[1].max_num_neighbors == priors[1].max_num_neighbors
