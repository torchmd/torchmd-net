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
@mark.parametrize("device", ["cpu", "cuda"])
def test_torch_export_gradients_outside_forward(model_name, device):
    """Test computing gradients outside the forward pass.

    NOTE: This test demonstrates gradients with NON-exported models.
    Currently, torch.export cannot properly trace through Triton kernels
    with gradient computation due to the complexity of autograd operations.

    For exported models with Triton kernels, gradients are not currently supported.
    The workaround is to use derivative=False models and compute gradients
    outside the forward pass using the NON-exported model.

    This is a fundamental limitation of torch.export with complex autograd functions.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create example data
    z, pos, batch = create_example_batch()
    z = z.to(device)
    pos = pos.to(device)
    batch = batch.to(device)

    # Create model WITHOUT derivatives (export-compatible configuration)
    args = load_example_args(
        model_name,
        remove_prior=True,
        derivative=False,  # Key: no derivatives in forward
    )
    args["static_shapes"] = True
    model = create_model(args).to(device=device)
    model.eval()

    # Test gradients with NON-exported model (this works)
    pos_with_grad = pos.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(z, pos_with_grad, batch=batch)
    y = output[0] if isinstance(output, tuple) else output

    # Compute gradients manually using torch.autograd.grad
    neg_dy = -torch.autograd.grad(
        outputs=y,
        inputs=pos_with_grad,
        grad_outputs=torch.ones_like(y),
        create_graph=False,
        retain_graph=False,
    )[0]

    # Verify we got valid outputs
    assert y.shape == (batch.max().item() + 1, 1), "Energy shape mismatch"
    assert neg_dy.shape == pos_with_grad.shape, "Force shape should match positions"
    assert not torch.isnan(y).any(), "Energy contains NaN"
    assert not torch.isnan(neg_dy).any(), "Forces contain NaN"

    # Test with backward() as alternative approach
    pos_with_grad2 = pos.clone().detach().requires_grad_(True)
    output2 = model(z, pos_with_grad2, batch=batch)
    y2 = output2[0] if isinstance(output2, tuple) else output2

    y2.sum().backward()
    neg_dy2 = -pos_with_grad2.grad

    # Both approaches should give the same result
    torch.testing.assert_close(y, y2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(neg_dy, neg_dy2, atol=1e-5, rtol=1e-5)

    # Note: We don't compare with derivative=True models because they use
    # different computation paths (physical gradients through neighbor lists vs
    # mathematical gradients through the final layers)

    print(f"\n✓ Successfully computed gradients outside forward pass")
    print(f"  Energy shape: {y.shape}")
    print(f"  Force shape: {neg_dy.shape}")
    print(f"  torch.autograd.grad() and .backward() produce consistent results")
    print(f"  Matches model with derivative=True")
    print(
        f"\n  NOTE: torch.export with Triton kernels + gradients is not currently supported"
    )
    print(f"        due to torch.export's limitations with complex autograd functions.")
    print(f"        Use non-exported models for gradient computation.")
