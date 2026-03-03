"""Tests that warp-ops and pure-Python TensorNet paths produce identical results.

Parametrized over AceFF-1.0, AceFF-1.1, AceFF-2.0 and cpu/cuda devices.
"""

import pytest
import torch
from torch.testing import assert_close

import torchmdnet.models.tensornet as _tn
import torchmdnet.models.tensornet2 as _tn2

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "aceff1.0": dict(repo_id="Acellera/AceFF-1.0", filename="aceff_v1.0.ckpt"),
    "aceff1.1": dict(repo_id="Acellera/AceFF-1.1", filename="aceff_v1.1.ckpt"),
    "aceff2.0": dict(repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_key, device):
    from huggingface_hub import hf_hub_download
    from torchmdnet.models.model import load_model

    cfg = MODELS[model_key]
    ckpt = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"])
    model = load_model(ckpt, derivative=True)
    return model.to(device)


def _caffeine_tensors(device, dtype=torch.float32):
    from ase.io import read
    from ase.data import atomic_numbers
    import os

    pdb = os.path.join(
        os.path.dirname(__file__), "../examples/aceff_examples/caffeine.pdb"
    )
    atoms = read(os.path.abspath(pdb))
    z = torch.tensor(
        [atomic_numbers[s] for s in atoms.get_chemical_symbols()], dtype=torch.long
    ).to(device)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype).to(device)
    return z, pos


def _run(model, z, pos):
    energy, forces = model(z, pos)
    return energy.detach(), forces.detach()


def _patch_nonopt(monkeypatch):
    """Switch both tensornet modules to pure-Python ops (OPT=False)."""
    monkeypatch.setattr(_tn, "OPT", False)
    monkeypatch.setattr(_tn, "compose_tensor", _tn._compose_tensor)
    monkeypatch.setattr(_tn, "decompose_tensor", _tn._decompose_tensor)
    monkeypatch.setattr(_tn, "tensor_matmul_o3", _tn._tensor_matmul_o3)
    monkeypatch.setattr(_tn, "tensor_matmul_so3", _tn._tensor_matmul_so3)
    monkeypatch.setattr(_tn2, "OPT", False)
    monkeypatch.setattr(_tn2, "compose_tensor", _tn._compose_tensor)
    monkeypatch.setattr(_tn2, "decompose_tensor", _tn._decompose_tensor)
    monkeypatch.setattr(_tn2, "tensor_matmul_o3", _tn._tensor_matmul_o3)
    monkeypatch.setattr(_tn2, "tensor_matmul_so3", _tn._tensor_matmul_so3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", list(MODELS))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warp_vs_python(model_key, device, monkeypatch):
    """Warp-ops and pure-Python paths must produce identical energy and forces."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _tn.OPT:
        pytest.skip("warp-ops not available; skipping OPT=True side of test")

    model = _load_model(model_key, device)
    z, pos = _caffeine_tensors(device)

    energy_opt, forces_opt = _run(model, z, pos)

    _patch_nonopt(monkeypatch)
    energy_py, forces_py = _run(model, z, pos)

    assert_close(energy_opt, energy_py, rtol=1e-4, atol=1e-4)
    assert_close(forces_opt, forces_py, rtol=1e-4, atol=1e-4)
