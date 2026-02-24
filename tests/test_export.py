# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pytest
from pytest import mark
import torch
from torchmdnet import models
from torchmdnet.models.model import create_model

from utils import load_example_args, create_example_batch


@mark.parametrize("model_name", ["tensornet"])
@mark.parametrize("device", ["cpu", "cuda"])
def test_torch_export(model_name, device):
    """Test that a model can be exported using torch.export and produces correct outputs.

    Note: This test requires derivative=False to avoid issues with requires_grad_ in the export.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # torch.export was introduced in PyTorch 2.1+
    if not hasattr(torch, "export"):
        pytest.skip("torch.export not available in this PyTorch version")

    # Create example data
    z, pos, batch = create_example_batch()
    z = z.to(device)
    pos = pos.to(device)
    batch = batch.to(device)

    # Create model with export-compatible settings
    args = load_example_args(
        model_name,
        remove_prior=True,
        derivative=False,  # Avoid requires_grad_ inside forward
    )
    # Add parameters after loading since they're not in the YAML config
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Get reference output from original model
    with torch.no_grad():
        y_ref = model(z, pos, batch=batch)

    # Export the model
    exported_program = torch.export.export(
        model, args=(z, pos), kwargs={"batch": batch}
    )

    # Run exported model
    with torch.no_grad():
        y_exported = exported_program.module()(z, pos, batch=batch)

    # Verify outputs match
    torch.testing.assert_close(y_ref, y_exported, atol=1e-5, rtol=1e-5)


@mark.parametrize("model_name", ["tensornet"])
def test_torch_export_dynamic_shapes(model_name):
    """Test torch.export with dynamic shapes across different atom counts.

    This test requires CUDA with Triton for the neighbor list computation.
    The Triton kernels are registered as custom ops via @triton_op, making them
    compatible with torch.export's symbolic tracing (no tril_indices needed).
    static_shapes=True avoids resize_to_fit boolean masking (data-dependent shapes).
    num_systems=1 avoids the data-dependent batch.max().item() call.
    """
    if not hasattr(torch, "export"):
        pytest.skip("torch.export not available in this PyTorch version")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA required for torch.export with dynamic shapes (Triton neighbor list)"
        )

    device = "cuda"

    # Create model with static_shapes=True so resize_to_fit=False
    # (avoids data-dependent boolean masking on neighbor list output)
    args = load_example_args(model_name, remove_prior=True, derivative=False)
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Create initial batch for export (single molecule)
    z, pos, batch = create_example_batch(n_atoms=6, multiple_batches=False)
    z, pos, batch = z.to(device), pos.to(device), batch.to(device)

    from torch.export import Dim

    # Define dynamic dimension for the number of atoms
    num_atoms_dim = Dim("num_atoms")
    dynamic_shapes = {
        "z": {0: num_atoms_dim},
        "pos": {0: num_atoms_dim},
        "batch": {0: num_atoms_dim},
        "num_systems": None,
    }

    # Pass num_systems=1 to avoid the data-dependent batch.max().item() call
    # during tracing, enabling torch.export without a warm-up step.
    exported_program = torch.export.export(
        model,
        args=(z, pos),
        kwargs={"batch": batch, "num_systems": 1},
        dynamic_shapes=dynamic_shapes,
    )

    # Test with different atom counts (single molecule each)
    for n_atoms in [3, 6, 12]:
        z_test, pos_test, batch_test = create_example_batch(
            n_atoms=n_atoms, multiple_batches=False
        )
        z_test = z_test.to(device)
        pos_test = pos_test.to(device)
        batch_test = batch_test.to(device)

        with torch.no_grad():
            y_ref = model(z_test, pos_test, batch=batch_test, num_systems=1)
            y_exported = exported_program.module()(
                z_test, pos_test, batch=batch_test, num_systems=1
            )

        torch.testing.assert_close(y_ref, y_exported, atol=1e-5, rtol=1e-5)


@mark.parametrize("model_name", ["tensornet"])
def test_torch_export_save_load(model_name, tmp_path):
    """Test that an exported model with dynamic shapes can be saved and loaded.

    Uses torch.export.save/load to serialize the ExportedProgram to disk,
    then verifies the loaded program produces correct outputs for different atom counts.
    """
    if not hasattr(torch, "export"):
        pytest.skip("torch.export not available in this PyTorch version")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA required for torch.export with dynamic shapes (Triton neighbor list)"
        )

    device = "cuda"

    # Create model
    args = load_example_args(model_name, remove_prior=True, derivative=False)
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Export with dynamic shapes
    z, pos, batch = create_example_batch(n_atoms=6, multiple_batches=False)
    z, pos, batch = z.to(device), pos.to(device), batch.to(device)

    from torch.export import Dim

    num_atoms_dim = Dim("num_atoms")
    dynamic_shapes = {
        "z": {0: num_atoms_dim},
        "pos": {0: num_atoms_dim},
        "batch": {0: num_atoms_dim},
        "num_systems": None,
    }

    exported_program = torch.export.export(
        model,
        args=(z, pos),
        kwargs={"batch": batch, "num_systems": 1},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # Save and load
    save_path = tmp_path / "exported_model.pt2"
    torch.export.save(exported_program, save_path)
    loaded_program = torch.export.load(save_path)

    # Test with different atom counts
    for n_atoms in [3, 6, 12]:
        z_test, pos_test, batch_test = create_example_batch(
            n_atoms=n_atoms, multiple_batches=False
        )
        z_test = z_test.to(device)
        pos_test = pos_test.to(device)
        batch_test = batch_test.to(device)

        with torch.no_grad():
            y_ref = model(z_test, pos_test, batch=batch_test, num_systems=1)
            y_loaded = loaded_program.module()(
                z_test, pos_test, batch=batch_test, num_systems=1
            )

        torch.testing.assert_close(y_ref, y_loaded, atol=1e-5, rtol=1e-5)


@mark.parametrize("model_name", ["tensornet"])
@mark.parametrize("device", ["cpu", "cuda"])
def test_torch_export_then_compile(model_name, device):
    """Test that an exported model can be torch.compiled afterwards for additional optimization."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not hasattr(torch, "export"):
        pytest.skip("torch.export not available in this PyTorch version")

    # Create example data
    z, pos, batch = create_example_batch()
    z = z.to(device)
    pos = pos.to(device)
    batch = batch.to(device)

    # Create and export model
    args = load_example_args(model_name, remove_prior=True, derivative=False)
    # Add parameters after loading since they're not in the YAML config
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Get reference output
    with torch.no_grad():
        y_ref = model(z, pos, batch=batch)

    # Export the model
    exported_program = torch.export.export(
        model, args=(z, pos), kwargs={"batch": batch}
    )

    # Now compile the exported model
    compiled_model = torch.compile(exported_program.module(), backend="inductor")

    with torch.no_grad():
        y_compiled = compiled_model(z, pos, batch=batch)

    # Verify outputs match
    torch.testing.assert_close(y_ref, y_compiled, atol=1e-5, rtol=1e-5)


@mark.parametrize("model_name", ["tensornet"])
def test_torch_export_gradients(model_name):
    """Test computing gradients (forces) through an exported model.

    Exports a derivative=False model with dynamic shapes, then computes
    forces as -dE/dpos using torch.autograd.grad on the exported model's output.
    Verifies that the exported model's gradients match the original model.
    """
    if not hasattr(torch, "export"):
        pytest.skip("torch.export not available in this PyTorch version")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA required for torch.export with dynamic shapes (Triton neighbor list)"
        )

    device = "cuda"

    # Create model
    args = load_example_args(model_name, remove_prior=True, derivative=False)
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Export with dynamic shapes
    z, pos, batch = create_example_batch(n_atoms=6, multiple_batches=False)
    z, pos, batch = z.to(device), pos.to(device), batch.to(device)

    from torch.export import Dim

    num_atoms_dim = Dim("num_atoms")
    dynamic_shapes = {
        "z": {0: num_atoms_dim},
        "pos": {0: num_atoms_dim},
        "batch": {0: num_atoms_dim},
        "num_systems": None,
    }

    exported_program = torch.export.export(
        model,
        args=(z, pos),
        kwargs={"batch": batch, "num_systems": 1},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # Test gradients with different atom counts
    for n_atoms in [3, 6, 12]:
        z_test, pos_test, batch_test = create_example_batch(
            n_atoms=n_atoms, multiple_batches=False
        )
        z_test = z_test.to(device)
        pos_test = pos_test.to(device)
        batch_test = batch_test.to(device)

        # Gradients through exported model
        pos_grad = pos_test.clone().detach().requires_grad_(True)
        y_exported = exported_program.module()(
            z_test, pos_grad, batch=batch_test, num_systems=1
        )
        energy_exported = y_exported[0] if isinstance(y_exported, tuple) else y_exported
        forces_exported = -torch.autograd.grad(energy_exported.sum(), pos_grad)[0]

        # Gradients through original model
        pos_grad_ref = pos_test.clone().detach().requires_grad_(True)
        y_ref = model(z_test, pos_grad_ref, batch=batch_test, num_systems=1)
        energy_ref = y_ref[0] if isinstance(y_ref, tuple) else y_ref
        forces_ref = -torch.autograd.grad(energy_ref.sum(), pos_grad_ref)[0]

        # Verify match
        torch.testing.assert_close(energy_exported, energy_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(forces_exported, forces_ref, atol=1e-5, rtol=1e-5)
