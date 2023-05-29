import os
import torch
import numpy as np
from torchmdnet.models.utils import Distance, OptimizedDistance


def benchmark_neighbors(
    device, strategy, n_batches, total_num_particles, mean_num_neighbors, density
):
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
    mean_num_neighbors : int
        Mean number of neighbors per particle.
    density : float
        Density of the system.
    Returns
    -------
    float
        Average time per batch in seconds.
    """
    torch.random.manual_seed(12344)
    np.random.seed(43211)
    num_particles = total_num_particles // n_batches
    expected_num_neighbors = mean_num_neighbors
    cutoff = np.cbrt(3 * expected_num_neighbors / (4 * np.pi * density))
    n_atoms_per_batch = torch.randint(
        int(num_particles / 2), int(num_particles * 2), size=(n_batches,), device="cpu"
    )
    # Fix so that the total number of particles is correct. Special care if the difference is negative
    difference = total_num_particles - n_atoms_per_batch.sum()
    if difference > 0:
        while difference > 0:
            i = np.random.randint(0, n_batches)
            n_atoms_per_batch[i] += 1
            difference -= 1
    else:
        while difference < 0:
            i = np.random.randint(0, n_batches)
            if n_atoms_per_batch[i] > num_particles:
                n_atoms_per_batch[i] -= 1
                difference += 1
    lbox = np.cbrt(num_particles / density)
    batch = torch.repeat_interleave(
        torch.arange(n_batches, dtype=torch.int64), n_atoms_per_batch
    ).to(device)
    cumsum = np.cumsum(np.concatenate([[0], n_atoms_per_batch]))
    pos = torch.rand(cumsum[-1], 3, device="cpu").to(device) * lbox
    if strategy != "distance":
        max_num_pairs = (expected_num_neighbors * n_atoms_per_batch.sum()).item() * 2
        box = torch.eye(3, device=device) * lbox
        nl = OptimizedDistance(
            cutoff_upper=cutoff,
            max_num_pairs=max_num_pairs,
            strategy=strategy,
            box=box,
            loop=False,
            include_transpose=True,
            check_errors=False,
            resize_to_fit=False,
        )
    else:
        max_num_neighbors = int(expected_num_neighbors * 5)
        nl = Distance(
            loop=False,
            cutoff_lower=0.0,
            cutoff_upper=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            neighbors, distances, distance_vecs = nl(pos, batch)
    torch.cuda.synchronize()
    nruns = 50
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph = torch.cuda.CUDAGraph()
    # record in a cuda graph
    if strategy != "distance":
        with torch.cuda.graph(graph):
            neighbors, distances, distance_vecs = nl(pos, batch)
        start.record()
        for i in range(nruns):
            graph.replay()
        end.record()
    else:
        start.record()
        for i in range(nruns):
            neighbors, distances, distance_vecs = nl(pos, batch)
        end.record()
    torch.cuda.synchronize()
    # Final time
    return start.elapsed_time(end) / nruns


if __name__ == "__main__":
    n_particles = 32767
    mean_num_neighbors = min(n_particles, 64)
    density = 0.8
    print(
        "Benchmarking neighbor list generation for {} particles with {} neighbors on average".format(
            n_particles, mean_num_neighbors
        )
    )
    results = {}
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for strategy in ["shared", "brute", "cell", "distance"]:
        for n_batches in batch_sizes:
            time = benchmark_neighbors(
                device="cuda",
                strategy=strategy,
                n_batches=n_batches,
                total_num_particles=n_particles,
                mean_num_neighbors=mean_num_neighbors,
                density=density,
            )
            # Store results in a dictionary
            results[strategy, n_batches] = time
    print("Summary")
    print("-------")
    print(
        "{:<10} {:<21} {:<21} {:<18} {:<10}".format(
            "Batch size", "Shared(ms)", "Brute(ms)", "Cell(ms)", "Distance(ms)"
        )
    )
    print(
        "{:<10} {:<21} {:<21} {:<18} {:<10}".format(
            "----------", "---------", "---------", "---------", "---------"
        )
    )
    # Print a column per strategy, show speedup over Distance in parenthesis
    for n_batches in batch_sizes:
        base = results["distance", n_batches]
        print(
            "{:<10} {:<4.2f} x{:<14.2f} {:<4.2f} x{:<14.2f}  {:<4.2f} x{:<14.2f} {:<10.2f}".format(
                n_batches,
                results["shared", n_batches],
                base / results["shared", n_batches],
                results["brute", n_batches],
                base / results["brute", n_batches],
                results["cell", n_batches],
                base / results["cell", n_batches],
                results["distance", n_batches],
            )
        )
    n_particles_list = np.power(2, np.arange(8, 18))

    for n_batches in [1, 2, 32, 64]:
        print(
            "Benchmarking neighbor list generation for {} batches with {} neighbors on average".format(
                n_batches, mean_num_neighbors
            )
        )
        results = {}
        for strategy in ["shared", "brute", "cell", "distance"]:
            for n_particles in n_particles_list:
                mean_num_neighbors = min(n_particles, 64)
                time = benchmark_neighbors(
                    device="cuda",
                    strategy=strategy,
                    n_batches=n_batches,
                    total_num_particles=n_particles,
                    mean_num_neighbors=mean_num_neighbors,
                    density=density,
                )
                # Store results in a dictionary
                results[strategy, n_particles] = time
        print("Summary")
        print("-------")
        print(
            "{:<10} {:<21} {:<21} {:<18} {:<10}".format(
                "N Particles", "Shared(ms)", "Brute(ms)", "Cell(ms)", "Distance(ms)"
            )
        )
        print(
            "{:<10} {:<21} {:<21} {:<18} {:<10}".format(
                "----------", "---------", "---------", "---------", "---------"
            )
        )
        # Print a column per strategy, show speedup over Distance in parenthesis
        for n_particles in n_particles_list:
            base = results["distance", n_particles]
            brute_speedup = base / results["brute", n_particles]
            if n_particles > 32000:
                results["brute", n_particles] = 0
                brute_speedup = 0
            print(
                "{:<10} {:<4.2f} x{:<14.2f} {:<4.2f} x{:<14.2f}  {:<4.2f} x{:<14.2f} {:<10.2f}".format(
                    n_particles,
                    results["shared", n_particles],
                    base / results["shared", n_particles],
                    results["brute", n_particles],
                    brute_speedup,
                    results["cell", n_particles],
                    base / results["cell", n_particles],
                    results["distance", n_particles],
                )
            )
