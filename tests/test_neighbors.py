import os
import pytest
import torch
import torch.jit
import numpy as np
from torchmdnet.models.utils import OptimizedDistance

def sort_neighbors(neighbors, deltas, distances):
    i_sorted = np.lexsort(neighbors)
    return neighbors[:, i_sorted], deltas[i_sorted], distances[i_sorted]


def apply_pbc(deltas, box_vectors):
    if box_vectors is None:
        return deltas
    else:
        ref_vectors = box_vectors.cpu().detach().numpy()
        deltas -= np.outer(np.round(deltas[:, 2] / ref_vectors[2, 2]), ref_vectors[2])
        deltas -= np.outer(np.round(deltas[:, 1] / ref_vectors[1, 1]), ref_vectors[1])
        deltas -= np.outer(np.round(deltas[:, 0] / ref_vectors[0, 0]), ref_vectors[0])
        return deltas


def compute_ref_neighbors(pos, batch, loop, include_transpose, cutoff, box_vectors):
    batch = batch.cpu()
    n_atoms_per_batch = torch.bincount(batch)
    n_batches = n_atoms_per_batch.shape[0]
    cumsum = (
        torch.cumsum(torch.cat([torch.tensor([0]), n_atoms_per_batch]), dim=0)
        .cpu()
        .detach()
        .numpy()
    )
    ref_neighbors = np.concatenate(
        [
            np.tril_indices(int(n_atoms_per_batch[i]), -1) + cumsum[i]
            for i in range(n_batches)
        ],
        axis=1,
    )
    # add the upper triangle
    if include_transpose:
        ref_neighbors = np.concatenate(
            [ref_neighbors, np.flip(ref_neighbors, axis=0)], axis=1
        )
    if loop:  # Add self interactions
        ilist = np.arange(cumsum[-1])
        ref_neighbors = np.concatenate(
            [ref_neighbors, np.stack([ilist, ilist], axis=0)], axis=1
        )
    pos_np = pos.cpu().detach().numpy()
    ref_distance_vecs = apply_pbc(
        pos_np[ref_neighbors[0]] - pos_np[ref_neighbors[1]], box_vectors
    )
    ref_distances = np.linalg.norm(ref_distance_vecs, axis=-1)

    # remove pairs with distance > cutoff
    mask = ref_distances < cutoff
    ref_neighbors = ref_neighbors[:, mask]
    ref_distance_vecs = ref_distance_vecs[mask]
    ref_distances = ref_distances[mask]
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(
        ref_neighbors, ref_distance_vecs, ref_distances
    )
    return ref_neighbors, ref_distance_vecs, ref_distances


@pytest.mark.parametrize(("device", "strategy"), [("cpu", "brute"), ("cuda", "brute"), ("cuda", "shared"), ("cuda", "cell")])
@pytest.mark.parametrize("n_batches", [1, 2, 3, 4, 128])
@pytest.mark.parametrize("cutoff", [0.1, 1.0, 3.0, 4.9])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neighbors(
    device, strategy, n_batches, cutoff, loop, include_transpose, box_type, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic not supported for cell")
    if device == "cpu" and strategy != "brute":
        pytest.skip("Only brute force supported on CPU")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    lbox = 10.0
    pos = torch.rand(cumsum[-1], 3, device=device, dtype=dtype) * lbox - 10.0*lbox
    # Ensure there is at least one pair
    pos[0, :] = torch.zeros(3)
    pos[1, :] = torch.zeros(3)
    pos.requires_grad = True
    if box_type is None:
        box = None
    elif box_type == "rectangular":
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(pos.dtype)
            .to(device)
        )
    elif box_type == "triclinic":
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.1, lbox, 0.0], [0.3, 0.2, lbox]])
            .to(pos.dtype)
            .to(device)
        )
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(
        pos, batch, loop, include_transpose, cutoff, box
    )
    max_num_pairs = ref_neighbors.shape[1]
    nl = OptimizedDistance(
        cutoff_lower=0.0,
        loop=loop,
        cutoff_upper=cutoff,
        max_num_pairs=max_num_pairs,
        strategy=strategy,
        box=box,
        return_vecs=True,
        include_transpose=include_transpose,
    )
    batch.to(device)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(
        neighbors, distance_vecs, distances
    )
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)

    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)

@pytest.mark.parametrize(("device", "strategy"), [("cpu", "brute"), ("cuda", "brute"), ("cuda", "shared"), ("cuda", "cell")])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_atoms", [1, 2, 3, 5, 100, 1000])
@pytest.mark.parametrize("grad", ["deltas", "distances", "combined"])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
def test_neighbor_grads(
    device, strategy, loop, include_transpose, dtype, num_atoms, grad, box_type
):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("No GPU")
    if device == "cpu" and strategy != "brute":
        pytest.skip("Only brute force supported on CPU")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    cutoff = 4.999999
    lbox = 10.0
    torch.random.manual_seed(1234)
    np.random.seed(123456)
    # Generate random positions
    positions = 0.25 * lbox * torch.rand(num_atoms, 3, device=device, dtype=dtype)
    if box_type is None:
        box = None
    else:
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(dtype)
            .to(device)
        )
    # Compute reference values using pure pytorch
    ref_neighbors = torch.vstack(
        (torch.tril_indices(num_atoms, num_atoms, -1, device=device),)
    )
    if include_transpose:
        ref_neighbors = torch.hstack(
            (ref_neighbors, torch.stack((ref_neighbors[1], ref_neighbors[0])))
        )
    if loop:
        index = torch.arange(num_atoms, device=device)
        ref_neighbors = torch.hstack((ref_neighbors, torch.stack((index, index))))
    ref_positions = positions.clone()
    ref_positions.requires_grad_(True)
    # Every pair is included, so there is no need to filter out pairs even after PBC
    ref_deltas = ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]]
    if box is not None:
        ref_box = box.clone()
        ref_deltas -= torch.outer(
            torch.round(ref_deltas[:, 2] / ref_box[2, 2]), ref_box[2]
        )
        ref_deltas -= torch.outer(
            torch.round(ref_deltas[:, 1] / ref_box[1, 1]), ref_box[1]
        )
        ref_deltas -= torch.outer(
            torch.round(ref_deltas[:, 0] / ref_box[0, 0]), ref_box[0]
        )

    if loop:
        ref_distances = torch.zeros((ref_deltas.size(0),), device=device, dtype=dtype)
        mask = ref_neighbors[0] != ref_neighbors[1]
        ref_distances[mask] = torch.linalg.norm(ref_deltas[mask], dim=-1)
    else:
        ref_distances = torch.linalg.norm(ref_deltas, dim=-1)
    max_num_pairs = max(ref_neighbors.shape[1], 1)
    positions.requires_grad_(True)
    nl = OptimizedDistance(
        cutoff_upper=cutoff,
        max_num_pairs=max_num_pairs,
        strategy=strategy,
        loop=loop,
        include_transpose=include_transpose,
        return_vecs=True,
        resize_to_fit=True,
        box=box,
    )
    neighbors, distances, deltas = nl(positions)
    # Check neighbor pairs are correct
    ref_neighbors_sort, _, _ = sort_neighbors(
        ref_neighbors.clone().cpu().detach().numpy(),
        ref_deltas.clone().cpu().detach().numpy(),
        ref_distances.clone().cpu().detach().numpy(),
    )
    neighbors_sort, _, _ = sort_neighbors(
        neighbors.clone().cpu().detach().numpy(),
        deltas.clone().cpu().detach().numpy(),
        distances.clone().cpu().detach().numpy(),
    )
    assert np.allclose(ref_neighbors_sort, neighbors_sort)

    # Compute gradients
    if grad == "deltas":
        ref_deltas.sum().backward()
        deltas.sum().backward()
    elif grad == "distances":
        ref_distances.sum().backward()
        distances.sum().backward()
    elif grad == "combined":
        (ref_deltas.sum() + ref_distances.sum()).backward()
        (deltas.sum() + distances.sum()).backward()
    else:
        raise ValueError("grad")
    ref_pos_grad_sorted = ref_positions.grad.cpu().detach().numpy()
    pos_grad_sorted = positions.grad.cpu().detach().numpy()
    if dtype == torch.float32:
        assert np.allclose(ref_pos_grad_sorted, pos_grad_sorted, atol=1e-2, rtol=1e-2)
    else:
        assert np.allclose(ref_pos_grad_sorted, pos_grad_sorted, atol=1e-8, rtol=1e-5)

@pytest.mark.parametrize(("device", "strategy"), [("cpu", "brute"), ("cuda", "brute"), ("cuda", "shared"), ("cuda", "cell")])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("num_atoms", [1,2,10])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
def test_neighbor_autograds(
    device, strategy, loop, include_transpose, num_atoms, box_type
):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("No GPU")
    if device == "cpu" and strategy != "brute":
        pytest.skip("Only brute force supported on CPU")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    dtype = torch.float64
    cutoff = 4.999999
    lbox = 10.0
    torch.random.manual_seed(1234)
    np.random.seed(123456)
    # Generate random positions
    if box_type is None:
        box = None
    else:
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(dtype)
            .to(device)
        )
    nl = OptimizedDistance(
        cutoff_upper=cutoff,
        max_num_pairs=num_atoms * (num_atoms),
        strategy=strategy,
        loop=loop,
        include_transpose=include_transpose,
        return_vecs=True,
        resize_to_fit=True,
        box=box,
    )
    positions = 0.25 * lbox * torch.rand(num_atoms, 3, device=device, dtype=dtype)
    positions.requires_grad_(True)
    batch = torch.zeros((num_atoms,), dtype=torch.long, device=device)
    neighbors, distances, deltas = nl(positions, batch)
    # Lambda that returns only the distances and deltas
    lambda_dist = lambda x, y: nl(x, y)[1:]
    torch.autograd.gradcheck(lambda_dist, (positions, batch), eps=1e-4, atol=1e-4, rtol=1e-4, nondet_tol=1e-4)
    torch.autograd.gradgradcheck(lambda_dist, (positions, batch), eps=1e-4, atol=1e-4, rtol=1e-4, nondet_tol=1e-4)


@pytest.mark.parametrize("strategy", ["brute", "cell", "shared"])
@pytest.mark.parametrize("n_batches", [1, 2, 3, 4])
def test_large_size(strategy, n_batches):
    device = "cuda"
    cutoff = 1.76
    loop = False
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(4321)
    num_atoms = int(32000 / n_batches)
    n_atoms_per_batch = torch.ones(n_batches, dtype=torch.int64) * num_atoms
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    lbox = 45.0
    pos = torch.rand(cumsum[-1], 3, device=device) * lbox
    # Ensure there is at least one pair
    pos[0, :] = torch.zeros(3)
    pos[1, :] = torch.zeros(3)
    pos.requires_grad = True
    # Find the particle appearing in the most pairs
    max_num_neighbors = 64
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(
        pos, batch, loop, True, cutoff, None
    )
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(
        ref_neighbors, ref_distance_vecs, ref_distances
    )

    max_num_pairs = ref_neighbors.shape[1]

    # Must check without PBC since Distance does not support it
    box = None
    nl = OptimizedDistance(
        cutoff_lower=0.0,
        loop=loop,
        cutoff_upper=cutoff,
        max_num_pairs=max_num_pairs,
        strategy=strategy,
        box=box,
        return_vecs=True,
        include_transpose=True,
        resize_to_fit=True,
    )
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(
        neighbors, distance_vecs, distances
    )
    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)

@pytest.mark.parametrize(("device", "strategy"), [("cpu", "brute"), ("cuda", "brute"), ("cuda", "shared"), ("cuda", "cell")])
@pytest.mark.parametrize("n_batches", [1, 128])
@pytest.mark.parametrize("cutoff", [1.0])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_jit_script_compatible(
    device, strategy, n_batches, cutoff, loop, include_transpose, box_type, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    if device == "cpu" and strategy != "brute":
        pytest.skip("Only brute force supported on CPU")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    lbox = 10.0
    pos = torch.rand(cumsum[-1], 3, device=device, dtype=dtype) * lbox
    # Ensure there is at least one pair
    pos[0, :] = torch.zeros(3)
    pos[1, :] = torch.zeros(3)
    pos.requires_grad = True
    if box_type is None:
        box = None
    else:
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(pos.dtype)
            .to(device)
        )
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(
        pos, batch, loop, include_transpose, cutoff, box
    )
    max_num_pairs = ref_neighbors.shape[1]
    nl = OptimizedDistance(
        cutoff_lower=0.0,
        loop=loop,
        cutoff_upper=cutoff,
        max_num_pairs=max_num_pairs,
        strategy=strategy,
        box=box,
        return_vecs=True,
        include_transpose=include_transpose,
    )
    batch.to(device)

    nl = torch.jit.script(nl)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(
        neighbors, distance_vecs, distances
    )
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)

    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("strategy", ["brute", "shared", "cell"])
@pytest.mark.parametrize("n_batches", [1, 128])
@pytest.mark.parametrize("cutoff", [1.0])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cuda_graph_compatible_forward(
    device, strategy, n_batches, cutoff, loop, include_transpose, box_type, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    lbox = 10.0
    pos = torch.rand(cumsum[-1], 3, device=device, dtype=dtype) * lbox
    # Ensure there is at least one pair
    pos[0, :] = torch.zeros(3)
    pos[1, :] = torch.zeros(3)
    pos.requires_grad_(True)
    if box_type is None:
        box = None
    else:
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(pos.dtype)
            .to(device)
        )
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(
        pos, batch, loop, include_transpose, cutoff, box
    )
    max_num_pairs = ref_neighbors.shape[1]
    nl = OptimizedDistance(
        cutoff_lower=0.0,
        loop=loop,
        cutoff_upper=cutoff,
        max_num_pairs=max_num_pairs,
        strategy=strategy,
        box=box,
        return_vecs=True,
        include_transpose=include_transpose,
        check_errors=False,
        resize_to_fit=False,
    )
    batch.to(device)

    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    # Warm up
    with torch.cuda.stream(s):
        for _ in range(10):
            neighbors, distances, distance_vecs = nl(pos, batch)
    torch.cuda.synchronize()
    # Capture
    with torch.cuda.graph(graph):
        neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors.fill_(0)
    graph.replay()
    torch.cuda.synchronize()

    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(
        neighbors, distance_vecs, distances
    )
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)

    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)

@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("strategy", ["brute", "shared", "cell"])
@pytest.mark.parametrize("n_batches", [1, 128])
@pytest.mark.parametrize("cutoff", [1.0])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cuda_graph_compatible_backward(
    device, strategy, n_batches, cutoff, loop, include_transpose, box_type, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    lbox = 10.0
    pos = torch.rand(cumsum[-1], 3, device=device, dtype=dtype) * lbox
    # Ensure there is at least one pair
    pos[0, :] = torch.zeros(3)
    pos[1, :] = torch.zeros(3)
    pos.requires_grad_(True)
    if box_type is None:
        box = None
    else:
        box = (
            torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]])
            .to(pos.dtype)
            .to(device)
        )
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(
        pos, batch, loop, include_transpose, cutoff, box
    )
    max_num_pairs = ref_neighbors.shape[1]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        nl = OptimizedDistance(
            cutoff_lower=0.0,
            loop=loop,
            cutoff_upper=cutoff,
            max_num_pairs=max_num_pairs,
            strategy=strategy,
            box=box,
            return_vecs=True,
            include_transpose=include_transpose,
            check_errors=False,
            resize_to_fit=False,
        )
        batch.to(device)

        graph = torch.cuda.CUDAGraph()
        # Warm up
        neighbors, distappnces, distance_vecs = nl(pos, batch)
        for _ in range(10):
            neighbors, distances, distance_vecs = nl(pos, batch)
            distances.sum().backward()
            pos.grad.data.zero_()
        torch.cuda.synchronize()

        # Capture
        with torch.cuda.graph(graph):
            neighbors, distances, distance_vecs = nl(pos, batch)
            distances.sum().backward()
            pos.grad.data.zero_()
        graph.replay()
        torch.cuda.synchronize()
