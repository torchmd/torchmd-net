import os
import pytest
import torch
import numpy as np
from torch.cuda import check_error

from torchmdnet.neighbors import get_neighbor_pairs


class DistanceCellList(torch.nn.Module):
    def __init__(
        self,
        cutoff_upper,
        max_num_pairs=32,
        loop=False,
    ):
        super(DistanceCellList, self).__init__()
        """ Compute the neighbor list for a given cutoff.
        Parameters
        ----------
        cutoff_upper : float
            Upper cutoff for the neighbor list.
        max_num_pairs : int
            Maximum number of pairs to store.
        loop : bool
            Whether to include self interactions (pair (i,i)).
        """
        self.cutoff_upper = cutoff_upper
        self.max_num_pairs = max_num_pairs
        self.loop = loop

    def forward(self, pos, batch):
        """
        Parameters
        ----------
        pos : torch.Tensor
            shape (N, 3)
        batch : torch.Tensor
            shape (N,)
        Returns
        -------
        neighbors : torch.Tensor
          List of neighbors for each atom in the batch.
        shape (2, max_num_pairs)
        distances : torch.Tensor
            List of distances for each atom in the batch.
        shape (max_num_pairs,)
        distance_vecs : torch.Tensor
            List of distance vectors for each atom in the batch.
        shape (max_num_pairs, 3)

        """
        neighbors, distance_vecs, distances = get_neighbor_pairs(
            pos,
            cutoff=self.cutoff_upper,
            batch=batch,
            max_num_pairs=self.max_num_pairs,
            check_errors=True
        )
        return neighbors, distances, distance_vecs

def sort_neighbors(neighbors, deltas, distances):
    i_sorted = np.lexsort(neighbors)
    return neighbors[:, i_sorted], deltas[i_sorted], distances[i_sorted]

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n_batches", [1, 2, 3])
@pytest.mark.parametrize("cutoff", [0.1, 1.5, 1000.0])
def test_neighbors(device, n_batches, cutoff):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    n_atoms_per_batch = np.random.randint(2, 4, size=n_batches)
    batch = torch.tensor([i for i in range(n_batches) for j in range(n_atoms_per_batch[i])])
    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    pos = torch.randn(cumsum[-1], 3, device=device)
    #Ensure there is at least one pair
    pos[0,:] = torch.zeros(3)
    pos[1,:] = torch.zeros(3)
    pos.requires_grad = True
    print("Pos")
    print(pos.shape)
    print("Batch")
    print(batch)
    ref_neighbors = np.concatenate([np.tril_indices(n_atoms_per_batch[i], -1)+cumsum[i] for i in range(n_batches)], axis=1)
    print("Neighbors_i concat")
    print(ref_neighbors.shape)
    print(ref_neighbors)
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

    nl = DistanceCellList(cutoff_upper=cutoff, max_num_pairs=max_num_pairs)
    batch.to(device)
    neighbors, distances, distance_vecs = nl(pos, batch)
    neighbors = neighbors.cpu().detach().numpy()
    distance_vecs = distance_vecs.cpu().detach().numpy()
    distances = distances.cpu().detach().numpy()
    assert neighbors.shape == (2, max_num_pairs)
    assert distances.shape == (max_num_pairs,)
    assert distance_vecs.shape == (max_num_pairs, 3)

    print("Neighbors")
    print(neighbors)
    print(ref_neighbors)
    # print("Distances")
    # print(distances)
    # print(ref_distances)
    # print("Distance vecs")
    # print(distance_vecs)
    # print(ref_distance_vecs)
    neighbors, distance_vecs, distances = sort_neighbors(neighbors, distance_vecs, distances)


    assert np.allclose(neighbors, ref_neighbors)
    assert np.allclose(distances, ref_distances)
    assert np.allclose(distance_vecs, ref_distance_vecs)
