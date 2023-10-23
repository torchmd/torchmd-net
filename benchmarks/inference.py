"""
Benchmark script for inference.
This script compiles a model using torch.compile and measures the time it takes to run a forward and backward pass for a certain molecule.

"""
import torch
from os.path import dirname, join
import torch
from torchmdnet.models.model import create_model
import yaml
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable
import numpy as np
import time
import os


def load_example_args(model_name, remove_prior=False, config_file=None, **kwargs):
    if config_file is None:
        if model_name == "tensornet":
            config_file = join(
                dirname(dirname(__file__)), "examples", "TensorNet-QM9.yaml"
            )
        else:
            config_file = join(dirname(dirname(__file__)), "examples", "ET-QM9.yaml")
    with open(config_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args["model"] = model_name
    args["seed"] = 1234
    if remove_prior:
        args["prior_model"] = None
    for key, val in kwargs.items():
        assert key in args, f"Broken test! Unknown key '{key}'."
        args[key] = val
    return args


class GpuTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # synchronize CUDA device
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000  # Convert to milliseconds


def benchmark_pdb(pdb_file, **kwargs):
    device = "cuda"
    molecule = Molecule(pdb_file)
    atomic_numbers = torch.tensor(
        [periodictable[symbol].number for symbol in molecule.element],
        dtype=torch.long,
        device=device,
    )
    positions = torch.tensor(
        molecule.coords[:, :, 0], dtype=torch.float32, device=device
    ).to(device)
    molecule = None
    torch.cuda.nvtx.range_push("Initialization")
    args = load_example_args(
        "tensornet",
        config_file="../examples/TensorNet-rMD17.yaml",
        remove_prior=True,
        output_model="Scalar",
        derivative=False,
        max_z=int(atomic_numbers.max() + 1),
        max_num_neighbors=32,
        **kwargs,
    )
    model = create_model(args)
    z = atomic_numbers
    pos = positions
    batch = torch.zeros_like(z).to("cuda")
    model = model.to("cuda")
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Warmup")
    for i in range(3):
        pred, _ = model(z, pos, batch)
        pred.sum().backward()
    torch.cuda.synchronize()
    model = torch.compile(model, backend="inductor", mode="max-autotune")
    for i in range(10):
        pred, _ = model(z, pos, batch)
        pred.sum().backward()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Benchmark")
    nbench = 100
    times = np.zeros(nbench)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with GpuTimer() as timer:
        with torch.cuda.stream(stream):
            for i in range(nbench):
                # torch.cuda.synchronize()
                # with GpuTimer() as timer2:
                # torch.cuda.nvtx.range_push("Step")
                pred, _ = model(z, pos, batch)
                # torch.cuda.nvtx.range_push("derivative")
                pred.sum().backward()
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.synchronize()
                # times[i] = timer2.interval
        torch.cuda.synchronize()
    # torch.cuda.nvtx.range_pop()
    return len(atomic_numbers), timer.interval / nbench


from tabulate import tabulate

# List of cases to benchmark, arbitrary parameters can be overriden here
cases = {
    "0L": {"num_layers": 0, "embedding_dimension": 128},
    "1L": {"num_layers": 1, "embedding_dimension": 128},
    "2L": {"num_layers": 2, "embedding_dimension": 128},
    "2L emb 64": {"num_layers": 2, "embedding_dimension": 64},
}

def benchmark_all():
    timings = {}
    for pdb_file in os.listdir("systems"):
        if not pdb_file.endswith(".pdb"):
            continue
        molecule = Molecule(os.path.join("systems", pdb_file))
        natoms = len(molecule.element)
        print("Found %s, with %d atoms" % (pdb_file, natoms))
    for pdb_file in os.listdir("systems"):
        if not pdb_file.endswith(".pdb"):
            continue
        if pdb_file == "stmv.pdb":  # Does not fit in a 4090
            continue
        times = {}
        num_atoms = 0
        for name, kwargs in cases.items():
            torch.cuda.empty_cache()
            num_atoms, time = benchmark_pdb(os.path.join("systems", pdb_file), **kwargs)
            times[name] = time
        timings[pdb_file] = (num_atoms, times)
    # Print a table with the timings
    keys = list(timings.keys())
    values = list(timings.values())
    labels = [f"Time %s (ms)" % key for key in values[0][1].keys()]
    table_data = [["Molecule (atoms)", *labels]]
    for key, val in timings.items():
        # Remove the .pdb extension
        key = key[:-4]
        molecule = "{} ({})".format(key, val[0])
        times = []
        for time in val[1].values():
            times.append(round(time, 2))
        table_data.append([molecule, *times])
    table = tabulate(
        table_data,
        headers="firstrow",
        tablefmt="pretty",
        showindex=False,
        stralign="center",
        numalign="center",
        colalign=("center",),
    )
    print(table)


if __name__ == "__main__":
    benchmark_all()
