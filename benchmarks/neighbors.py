import os
import torch
import numpy as np
from torchmdnet.models.utils import DistanceCellList


def benchmark_neighbors(device, strategy, n_batches, total_num_particles, mean_num_neighbors=32):
    """Benchmark the neighbor list generation.

    Parameters
    ----------
    device : str
        Device to use for the benchmark.
    strategy : str
        Strategy to use for the neighbor list generation (cell, brute).
    n_batches : int
        Number of batches to generate.
    total_num_particles : int
        Total number of particles.
    Returns
    -------
    float
        Average time per batch in seconds.
    """
    density = 0.5;
    num_particles = total_num_particles // n_batches
    expected_num_neighbors = mean_num_neighbors
    cutoff = np.cbrt(3 * expected_num_neighbors / (4 * np.pi * density));
    n_atoms_per_batch = np.random.randint(num_particles-10, num_particles+10, size=n_batches)
    #Fix the last batch so that the total number of particles is correct
    n_atoms_per_batch[-1] += total_num_particles - n_atoms_per_batch.sum()
    if n_atoms_per_batch[-1] < 0:
        n_atoms_per_batch[-1] = 1

    lbox = np.cbrt(num_particles / density);
    batch = torch.repeat_interleave(torch.arange(n_batches, dtype=torch.int32, device=device), torch.tensor(n_atoms_per_batch, dtype=torch.int32, device=device))

    cumsum = np.cumsum( np.concatenate([[0], n_atoms_per_batch]))
    pos = torch.rand(cumsum[-1], 3, device=device)*lbox
    max_num_pairs = torch.tensor(expected_num_neighbors * n_atoms_per_batch.sum(), dtype=torch.int64).item()
    nl = DistanceCellList(cutoff_upper=cutoff, max_num_pairs=max_num_pairs, strategy=strategy, box=torch.Tensor([lbox, lbox, lbox]))
    #Warmup
    neighbors, distances, distance_vecs = nl(pos, batch)
    if device == 'cuda':
        torch.cuda.synchronize()
    #Benchmark using torch profiler
    nruns = 100
    if device == 'cuda':
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(nruns):
        neighbors, distances, distance_vecs = nl(pos, batch)
    end.record()
    if device == 'cuda':
        torch.cuda.synchronize()
    #Final time
    return (start.elapsed_time(end) / nruns)

if __name__ == '__main__':
    n_particles = 100000
    mean_num_neighbors = min(n_particles, 128);
    print("Benchmarking neighbor list generation for {} particles with {} neighbors on average".format(n_particles, mean_num_neighbors))
    for strategy in ['brute', 'cell']:
        print("Strategy: {}".format(strategy))
        print("--------")
        print("{:<10} {:<10}".format("Batch size", "Time (ms)"))
        print("{:<10} {:<10}".format("----------", "---------"))
        #Loop over different number of batches
        for n_batches in [1, 10, 100, 1000]:
            time = benchmark_neighbors(device='cuda',
                                       strategy=strategy,
                                       n_batches=n_batches,
                                       total_num_particles=n_particles,
                                       mean_num_neighbors=mean_num_neighbors
                                       )
            print("{:<10} {:<10.2f}".format(n_batches, time))
