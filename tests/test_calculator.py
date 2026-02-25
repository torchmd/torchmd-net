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


@pytest.mark.parametrize("box", [None, torch.eye(3)])
@pytest.mark.parametrize("use_cuda_graphs", [True, False])
def test_compare_forward(box, use_cuda_graphs):
    from copy import deepcopy

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
    # Patch the model
    model = deepcopy(model)
    model.representation_model.static_shapes = use_cuda_graphs
    model.representation_model.distance.resize_to_fit = not use_cuda_graphs
    calc_graph.model = model
    if box is not None:
        box = (box * 2 * args["cutoff_upper"]).unsqueeze(0)
    for _ in range(10):
        e_calc, f_calc = calc.calculate(pos, box)
        e_pred, f_pred = calc_graph.calculate(pos, box)
        assert_close(e_calc, e_pred)
        assert_close(f_calc, f_pred)


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

    assert_close(e_calc, e_pred)
    assert_close(f_calc, f_pred.view(-1, len(z1), 3), rtol=3e-4, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ase_calculator(device):
    import platform

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Skip on Windows CI for now because we don't have cl.exe installed for torch.compile
    if platform.system() == "Windows":
        pytest.skip("Skipping test on Windows")

    from torchmdnet.calculators import TMDNETCalculator
    from ase.io import read
    from ase import units
    from ase.md.langevin import Langevin
    import os
    import numpy as np

    ref_forces = np.array(
        [
            [-7.76999056e-01, -6.89736724e-01, -9.31625906e-03],
            [2.16895628e00, 5.29322922e-01, 9.77647374e-04],
            [-6.50327325e-01, 1.11337602e00, 2.27598846e-03],
            [1.19350255e00, -1.23314810e00, -7.83117674e-03],
            [5.11314452e-01, -3.33878160e-01, -5.03035402e-03],
            [-7.18148768e-01, 7.37230778e-02, 3.08941817e-03],
            [-1.25317931e-01, -5.19263268e-01, 2.00758013e-03],
            [-3.05806249e-01, -4.49415118e-01, -8.53991229e-03],
            [5.10734320e-03, 2.49908626e-01, 2.04431713e-02],
            [-3.65967184e-01, -1.57078415e-01, -1.55984145e-03],
            [-6.44133329e-01, 1.16345167e00, 4.54566162e-03],
            [2.05828249e-02, -2.64510632e-01, -1.38899162e-02],
            [1.73451304e-02, 3.65104795e-01, 1.13833081e-02],
            [-8.57830405e-01, -2.25283504e-01, -2.49589253e-02],
            [-1.56955227e-01, 1.19012646e-01, -1.87584094e-03],
            [-1.50042176e-02, 1.75106078e-02, 2.51995742e-01],
            [3.01239967e-01, 3.67318511e-01, 4.64916229e-06],
            [-9.57870483e-03, 1.21697336e-02, -2.39765823e-01],
            [-2.48186022e-01, 2.74000764e-02, -1.08634552e-03],
            [1.26295090e-01, 1.04473650e-01, 2.81187654e-01],
            [1.28753006e-01, 1.03064716e-01, -2.88918495e-01],
            [2.80321002e-01, -5.11180341e-01, -1.12308562e-03],
            [5.06305993e-02, 6.65888190e-02, -2.11322665e-01],
            [7.02065229e-02, 7.10679889e-02, 2.37307906e-01],
        ]
    )

    curr_dir = os.path.dirname(__file__)

    checkpoint = join(curr_dir, "example_tensornet.ckpt")
    calc = TMDNETCalculator(checkpoint, device=device)

    atoms = read(join(curr_dir, "caffeine.pdb"))
    atoms.calc = calc
    # The total molecular charge must be set
    atoms.info["charge"] = 0
    assert np.allclose(atoms.get_potential_energy(), -113.6652, atol=1e-4)
    assert np.allclose(atoms.get_forces(), ref_forces, atol=1e-4)

    # if we change the charge, the energy and forces should change
    atoms.info["charge"] = 100
    assert not np.allclose(atoms.get_potential_energy(), -113.6652, atol=1e-4)
    assert not np.allclose(atoms.get_forces(), ref_forces, atol=1e-4)

    # Molecular dynamics
    atoms.info["charge"] = 0

    temperature_K: float = 300
    timestep: float = 1.0 * units.fs
    friction: float = 0.01 / units.fs
    nsteps: int = 10
    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
    dyn.run(steps=nsteps)

    # Now we can do the same but enabling torch.compile for increased speed
    calc = TMDNETCalculator(checkpoint, device=device, compile=True)
    atoms.calc = calc
    # Run more dynamics
    dyn.run(steps=nsteps)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_aceff2_coulomb_cutoff(device):
    """Test AceFF-2 on caffeine (no PBC) with TMDNETCalculator.

    1. Compute energy with no coulomb_cutoff (all-to-all N^2 Coulomb).
    2. Compute energy with very large cutoff
    3. Compute energy with decreasing large coulomb_cutoffs (200 → 20 Å).
       Caffeine is ~10 Å across, so all pairs are within cutoff in every case.
       The Reaction Field correction is small at large cutoffs, so energies
       should be close to the no-cutoff reference.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub not available")

    try:
        model_file_path = hf_hub_download(
            repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"
        )
    except Exception:
        pytest.skip("Could not download AceFF-2.0 model from HuggingFace")

    from torchmdnet.calculators import TMDNETCalculator
    from ase.io import read
    import numpy as np
    import os

    curr_dir = os.path.dirname(__file__)
    caffeine_pdb = os.path.join(curr_dir, "caffeine.pdb")

    # --- reference: no coulomb_cutoff, all-to-all Coulomb, no PBC ---
    atoms_ref = read(caffeine_pdb)
    atoms_ref.info["charge"] = 0
    calc_ref = TMDNETCalculator(model_file_path, device=device)
    atoms_ref.calc = calc_ref
    energy_no_cutoff = atoms_ref.get_potential_energy()

    # in the limit of infinite cutoff the reaction field calculation should give the same energy
    calc_inf = TMDNETCalculator(
        model_file_path, device=device, coulomb_cutoff=1e6, coulomb_max_num_neighbors=20
    )
    atoms_inf = read(caffeine_pdb)
    atoms_inf.info["charge"] = 0
    atoms_inf.calc = calc_inf
    energy_inf = atoms_inf.get_potential_energy()
    np.testing.assert_allclose(
        energy_inf,
        energy_no_cutoff,
        rtol=1e-5,
        atol=1e-5,
    )

    # try changing the cutoff
    for cutoff in [2000.0, 200.0, 100.0, 50.0, 20.0]:
        atoms_c = read(caffeine_pdb)
        atoms_c.info["charge"] = 0
        calc_c = TMDNETCalculator(
            model_file_path,
            device=device,
            coulomb_cutoff=cutoff,
            coulomb_max_num_neighbors=64,
        )
        atoms_c.calc = calc_c
        energy_cutoff = atoms_c.get_potential_energy()
        # energy should increase as cutoff decreases as we are adding the reaction field energy
        assert energy_cutoff > energy_no_cutoff

        # the difference should be small
        assert (energy_cutoff - energy_no_cutoff) / energy_no_cutoff < 0.01


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_aceff2_pbc_vs_no_pbc(device):
    """Test AceFF-2 on explicit-solvent alanine-dipeptide with TMDNETCalculator.

    The PDB has a CRYST1 record (cell ~32.8 x 32.9 x 31.9 Å, 2269 atoms).

    1. Evaluate with PBC disabled – atoms see only intra-box neighbours.
    2. Evaluate with PBC enabled  – atoms see periodic images via the box.

    The two energies must differ, which proves that PBC is actually applied
    rather than silently ignored.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub not available")

    try:
        model_file_path = hf_hub_download(
            repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"
        )
    except Exception:
        pytest.skip("Could not download AceFF-2.0 model from HuggingFace")

    from torchmdnet.calculators import TMDNETCalculator
    from ase.io import read
    import numpy as np
    import os

    pdb_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "aceff_examples",
        "alanine-dipeptide-explicit.pdb",
    )
    coulomb_cutoff = 10.0

    # --- no PBC: disable the periodic box read from CRYST1 ---
    atoms_nopbc = read(pdb_path)
    atoms_nopbc.info["charge"] = 0
    atoms_nopbc.pbc = False
    calc_nopbc = TMDNETCalculator(
        model_file_path, device=device, coulomb_cutoff=coulomb_cutoff
    )
    atoms_nopbc.calc = calc_nopbc
    energy_nopbc = atoms_nopbc.get_potential_energy()
    print(f"No PBC: {energy_nopbc}")

    # --- with PBC: keep the cell and PBC flags from the CRYST1 record ---
    atoms_pbc = read(pdb_path)
    atoms_pbc.info["charge"] = 0
    assert atoms_pbc.pbc.all(), "Expected full PBC from CRYST1 record"
    calc_pbc = TMDNETCalculator(
        model_file_path, device=device, coulomb_cutoff=coulomb_cutoff
    )
    atoms_pbc.calc = calc_pbc
    energy_pbc = atoms_pbc.get_potential_energy()
    print(f"PBC: {energy_pbc}")

    # PBC adds periodic-image contributions to both the NN and Coulomb terms,
    # so the energies must be meaningfully different.
    assert not np.isclose(energy_pbc, energy_nopbc, rtol=1e-3), (
        f"PBC energy ({energy_pbc:.4f}) is too close to no-PBC energy ({energy_nopbc:.4f}); "
        "PBC may not be applied"
    )


if __name__ == "__main__":
    test_aceff2_coulomb_cutoff("cpu")
    test_aceff2_coulomb_cutoff("cuda")
    test_aceff2_pbc_vs_no_pbc("cpu")
    test_aceff2_pbc_vs_no_pbc("cuda")
