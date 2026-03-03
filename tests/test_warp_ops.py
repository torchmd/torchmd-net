"""Tests that warp-ops and pure-Python TensorNet paths produce identical results."""

import os
import pytest
import torch
from torch.testing import assert_close
from os.path import dirname, join

import torchmdnet.models.tensornet as _tn

CURR_DIR = dirname(__file__)
CKPT = join(CURR_DIR, "example_tensornet.ckpt")
CAFFEINE_PDB = join(CURR_DIR, "caffeine.pdb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(device):
    from torchmdnet.models.model import load_model

    return load_model(CKPT, derivative=True).to(device)


def _caffeine_tensors(device):
    from ase.io import read
    from ase.data import atomic_numbers

    atoms = read(CAFFEINE_PDB)
    z = torch.tensor(
        [atomic_numbers[s] for s in atoms.get_chemical_symbols()], dtype=torch.long
    ).to(device)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(device)
    return z, pos


def _run(model, z, pos):
    energy, forces = model(z, pos)
    return energy.detach(), forces.detach()


def _set_opt(model, value: bool):
    """Set .opt on the TensorNet representation model and all its submodules."""
    rep = model.representation_model
    rep.opt = value
    rep.tensor_embedding.opt = value
    for layer in rep.layers:
        layer.opt = value


def _patch_nonopt(monkeypatch, model):
    """Switch model to pure-Python ops by patching module-level ops and .opt flags."""
    # Module-level ops are used as globals inside forward() bodies, so they
    # still need to be swapped even though branching is now done via self.opt.
    monkeypatch.setattr(_tn, "compose_tensor", _tn._compose_tensor)
    monkeypatch.setattr(_tn, "decompose_tensor", _tn._decompose_tensor)
    monkeypatch.setattr(_tn, "tensor_matmul_o3", _tn._tensor_matmul_o3)
    monkeypatch.setattr(_tn, "tensor_matmul_so3", _tn._tensor_matmul_so3)
    _set_opt(model, False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warp_vs_python(device, monkeypatch):
    """Warp-ops and pure-Python paths must produce identical energy and forces."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _tn.OPT:
        pytest.skip("warp-ops not available")

    model = _load_model(device)
    z, pos = _caffeine_tensors(device)

    energy_opt, forces_opt = _run(model, z, pos)

    _patch_nonopt(monkeypatch, model)
    energy_py, forces_py = _run(model, z, pos)

    assert_close(energy_opt, energy_py, rtol=1e-4, atol=1e-4)
    assert_close(forces_opt, forces_py, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonopt_runs(device, monkeypatch):
    """Pure-Python (opt=False) path must produce finite energy and forces."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = _load_model(device)
    z, pos = _caffeine_tensors(device)
    _patch_nonopt(monkeypatch, model)

    energy, forces = _run(model, z, pos)

    assert torch.isfinite(energy).all(), "Energy contains non-finite values"
    assert torch.isfinite(forces).all(), "Forces contain non-finite values"
    assert forces.shape == pos.shape
