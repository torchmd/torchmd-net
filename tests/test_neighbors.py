import os
import pytest
import torch
import numpy as np
from torchmdnet.models.utils import DistanceCellList

def sort_neighbors(neighbors, deltas, distances):
    i_sorted = np.lexsort(neighbors)
    return neighbors[:, i_sorted], deltas[i_sorted], distances[i_sorted]

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("strategy", ["cell"])
@pytest.mark.parametrize("n_batches", [1])
@pytest.mark.parametrize("cutoff", [1])
def test_neighbors(device, strategy, n_batches, cutoff):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    np.random.seed(1234)
    torch.manual_seed(4321)
    n_atoms_per_batch = np.random.randint(2, 10, size=n_batches)
    batch = torch.tensor([i for i in range(n_batches) for j in range(n_atoms_per_batch[i])], device=device, dtype=torch.int)
    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    lbox=10.0
    pos = torch.rand(cumsum[-1], 3, device=device)*lbox
    #Ensure there is at least one pair
    pos[0,:] = torch.zeros(3)
    pos[1,:] = torch.zeros(3)
    pos.requires_grad = True
    print("batch")
    print(batch)
    print("pos")
    print(pos)

    ref_neighbors = np.concatenate([np.tril_indices(n_atoms_per_batch[i], -1)+cumsum[i] for i in range(n_batches)], axis=1)
    pos_np = pos.cpu().detach().numpy()
    ref_distances = np.linalg.norm(pos_np[ref_neighbors[0]] - pos_np[ref_neighbors[1]], axis=-1)
    ref_distance_vecs = pos_np[ref_neighbors[0]] - pos_np[ref_neighbors[1]]
    ref_neighbors, ref_distance_vecs, ref_distances = sort_neighbors(ref_neighbors, ref_distance_vecs, ref_distances)
    #remove pairs with distance > cutoff
    mask = ref_distances < cutoff
    ref_neighbors = ref_neighbors[:, mask]
    ref_distance_vecs = ref_distance_vecs[mask]
    ref_distances = ref_distances[mask]
    max_num_pairs = ref_neighbors.shape[1]
    box = torch.tensor([lbox, lbox, lbox])
    nl = DistanceCellList(cutoff_upper=cutoff, max_num_pairs=max_num_pairs, strategy=strategy, box=box)
    batch.to(device)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    print("neighbors")
    print(neighbors)
    print("ref_neighbors")
    print(ref_neighbors)
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)
    neighbors, distance_vecs, distances = sort_neighbors(neighbors, distance_vecs, distances)
    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)
