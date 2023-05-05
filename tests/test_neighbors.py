import os
import pytest
import torch
import numpy as np
from torchmdnet.models.utils import Distance, DistanceCellList

def sort_neighbors(neighbors, deltas, distances):
    i_sorted = np.lexsort(neighbors)
    return neighbors[:, i_sorted], deltas[i_sorted], distances[i_sorted]


def apply_pbc(deltas, box_vectors):
    if(box_vectors is None):
        return deltas
    else:
        ref_vectors = box_vectors.cpu().detach().numpy()
        deltas -= np.outer(np.round(deltas[:,2]/ref_vectors[2,2]), ref_vectors[2])
        deltas -= np.outer(np.round(deltas[:,1]/ref_vectors[1,1]), ref_vectors[1])
        deltas -= np.outer(np.round(deltas[:,0]/ref_vectors[0,0]), ref_vectors[0])
        return deltas


def compute_ref_neighbors(pos, batch, loop, include_transpose, cutoff, box_vectors):
    batch = batch.cpu()
    n_atoms_per_batch = torch.bincount(batch)
    n_batches = n_atoms_per_batch.shape[0]
    cumsum = torch.cumsum(torch.cat([torch.tensor([0]), n_atoms_per_batch]), dim=0).cpu().detach().numpy()
    ref_neighbors = np.concatenate([np.tril_indices(int(n_atoms_per_batch[i]), -1)+cumsum[i] for i in range(n_batches)], axis=1)
    # add the upper triangle
    if(include_transpose):
        ref_neighbors = np.concatenate([ref_neighbors, np.flip(ref_neighbors, axis=0)], axis=1)
    if(loop): # Add self interactions
        ilist=np.arange(cumsum[-1])
        ref_neighbors = np.concatenate([ref_neighbors, np.stack([ilist, ilist], axis=0)], axis=1)
    pos_np = pos.cpu().detach().numpy()
    ref_distance_vecs = apply_pbc(pos_np[ref_neighbors[0]] - pos_np[ref_neighbors[1]], box_vectors)
    ref_distances = np.linalg.norm(ref_distance_vecs, axis=-1)

    #remove pairs with distance > cutoff
    mask = ref_distances < cutoff
    ref_neighbors = ref_neighbors[:, mask]
    ref_distance_vecs = ref_distance_vecs[mask]
    ref_distances = ref_distances[mask]
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(ref_neighbors, ref_distance_vecs, ref_distances)
    return ref_neighbors, ref_distance_vecs, ref_distances

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("strategy", ["brute", "cell"])
@pytest.mark.parametrize("n_batches", [1, 2, 3, 4, 128])
@pytest.mark.parametrize("cutoff", [0.1, 1.0, 3.0, 4.9])
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("include_transpose", [True, False])
@pytest.mark.parametrize("box_type", [None, "triclinic", "rectangular"])
def test_neighbors(device, strategy, n_batches, cutoff, loop, include_transpose, box_type):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if box_type == "triclinic" and strategy == "cell":
        pytest.skip("Triclinic only supported for brute force")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch).to(device)
    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    lbox=10.0
    pos = torch.rand(cumsum[-1], 3, device=device)*lbox
    #Ensure there is at least one pair
    pos[0,:] = torch.zeros(3)
    pos[1,:] = torch.zeros(3)
    pos.requires_grad = True
    if(box_type is None):
        box = None
    else:
        box = torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]]).to(pos.dtype).to(device)
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(pos, batch, loop, include_transpose, cutoff, box)
    max_num_pairs = ref_neighbors.shape[1]
    nl = DistanceCellList(cutoff_lower=0.0, loop=loop, cutoff_upper=cutoff, max_num_pairs=max_num_pairs, strategy=strategy, box=box, return_vecs=True, include_transpose=include_transpose)
    batch.to(device)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(neighbors, distance_vecs, distances)
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)

    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("strategy", ["brute", "cell"])
@pytest.mark.parametrize("n_batches", [1, 2, 3, 4])
@pytest.mark.parametrize("cutoff", [0.1, 1.0, 1000.0])
@pytest.mark.parametrize("loop", [True, False])
def test_compatible_with_distance(device, strategy, n_batches, cutoff, loop):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(4321)
    n_atoms_per_batch = torch.randint(3, 100, size=(n_batches,))
    batch = torch.repeat_interleave(torch.arange(n_batches, dtype=torch.long), n_atoms_per_batch).to(device)
    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    lbox=10.0
    pos = torch.rand(cumsum[-1], 3, device=device)*lbox
    #Ensure there is at least one pair
    pos[0,:] = torch.zeros(3)
    pos[1,:] = torch.zeros(3)
    pos.requires_grad = True
    ref_neighbors, ref_distance_vecs, ref_distances = compute_ref_neighbors(pos, batch, loop, True, cutoff, None)
    #Find the particle appearing in the most pairs
    max_num_neighbors = torch.max(torch.bincount(torch.tensor(ref_neighbors[0,:])))
    d = Distance(cutoff_lower=0.0, cutoff_upper=cutoff, loop=loop, max_num_neighbors=max_num_neighbors, return_vecs=True)
    ref_neighbors, ref_distances, ref_distance_vecs = d(pos, batch)
    ref_neighbors = ref_neighbors.cpu().detach().numpy()
    ref_distance_vecs = ref_distance_vecs.cpu().detach().numpy()
    ref_distances = ref_distances.cpu().detach().numpy()
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(ref_neighbors, ref_distance_vecs, ref_distances)

    max_num_pairs = ref_neighbors.shape[1]
    box = None
    nl = DistanceCellList(cutoff_lower=0.0, loop=loop, cutoff_upper=cutoff, max_num_pairs=max_num_pairs, strategy=strategy, box=box, return_vecs=True, include_transpose=True)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(neighbors, distance_vecs, distances)
    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)



@pytest.mark.parametrize("strategy", ["brute", "cell"])
@pytest.mark.parametrize("n_batches", [1, 2, 3, 4])
def test_large_size(strategy, n_batches):
    device = "cuda"
    cutoff = 1.76
    loop = False
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(4321)
    num_atoms=int(32000/n_batches)
    n_atoms_per_batch = torch.ones(n_batches, dtype=torch.int64)*num_atoms
    batch = torch.repeat_interleave(torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch).to(device)
    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    lbox=45.0
    pos = torch.rand(cumsum[-1], 3, device=device)*lbox
    #Ensure there is at least one pair
    pos[0,:] = torch.zeros(3)
    pos[1,:] = torch.zeros(3)
    pos.requires_grad = True
    #Find the particle appearing in the most pairs
    max_num_neighbors = 64
    d = Distance(cutoff_lower=0.0, cutoff_upper=cutoff, loop=loop, max_num_neighbors=max_num_neighbors, return_vecs=True)
    ref_neighbors, ref_distances, ref_distance_vecs = d(pos, batch)
    ref_neighbors = ref_neighbors.cpu().detach().numpy()
    ref_distance_vecs = ref_distance_vecs.cpu().detach().numpy()
    ref_distances = ref_distances.cpu().detach().numpy()
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(ref_neighbors, ref_distance_vecs, ref_distances)

    max_num_pairs = ref_neighbors.shape[1]

    #Must check without PBC since Distance does not support it
    box = None #torch.tensor([[lbox, 0.0, 0.0], [0.0, lbox, 0.0], [0.0, 0.0, lbox]]).to(pos.dtype).to(device)
    nl = DistanceCellList(cutoff_lower=0.0, loop=loop, cutoff_upper=cutoff, max_num_pairs=max_num_pairs, strategy=strategy, box=box, return_vecs=True, include_transpose=True, resize_to_fit=True)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    neighbors, distance_vecs, distances = sort_neighbors(neighbors, distance_vecs, distances)
    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)
