import torch
import pytest
from pytest import mark
import pickle
from os.path import exists, dirname, join
import torch
import pytorch_lightning as pl
from torchmdnet import models
from torchmdnet.models.model import create_model
from torchmdnet.models import output_modules
import torch._dynamo as dynamo
import yaml
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable
import numpy as np
import time
def load_example_args(model_name, remove_prior=False, config_file=None, **kwargs):
    if config_file is None:
        if model_name == "tensornet":
            config_file = join(dirname(dirname(__file__)), "examples", "TensorNet-QM9.yaml")
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

def benchmark():
    print("Initializing")
    print(dynamo.list_backends())
    device = "cuda"
    pdb_file = "systems/alanine_dipeptide.pdb"
    molecule = Molecule(pdb_file)
    atomic_numbers = torch.tensor([periodictable[symbol].number for symbol in molecule.element], dtype=torch.long, device=device)
    positions = torch.tensor(molecule.coords[:,:,0], dtype=torch.float32, device=device).to(device)
    molecule = None
    print("Number of atoms: %d" % len(atomic_numbers))
    torch.cuda.nvtx.range_push("Initialization")
    # args = load_example_args(
    #     "equivariant-transformer",
    #     config_file="../examples/ET-MD17.yaml",
    #     remove_prior=True,
    #     output_model="Scalar",
    #     derivative=False,
    # )

    args = load_example_args(
        "tensornet",
        config_file="../examples/TensorNet-rMD17.yaml",
        remove_prior=True,
        output_model="Scalar",
        derivative=False,
    )
    model = create_model(args)

    z = atomic_numbers
    pos = positions
    batch = torch.zeros_like(z).to("cuda")
    model = model.to("cuda")
    torch.cuda.nvtx.range_pop()
    #Warmup
    torch.cuda.nvtx.range_push("Warmup")
    print("Warmup")
    #Count time
    for i in range(3):
        pred, _ = model(z, pos, batch)
        pred.sum().backward()
    torch.cuda.synchronize()
    model = torch.compile(model, backend="inductor", mode="max-autotune")
#    model.compile(z,pos,batch)
    for i in range(10):
        pred, _ = model(z, pos, batch)
        pred.sum().backward()

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Benchmark")
    print("Benchmark")
    #Profile with pytorch
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    nbench = 100
    times = np.zeros(nbench)
    stream = torch.cuda.Stream()
    with GpuTimer() as timer:
        with torch.cuda.stream(stream):
            for i in range(nbench):
                torch.cuda.synchronize()
                with GpuTimer() as timer2:
                    torch.cuda.nvtx.range_push("Step")
                    pred, _ = model(z, pos, batch)
                    torch.cuda.nvtx.range_push("derivative")
                    pred.sum().backward()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                times[i] = timer2.interval
            torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Time: %f ms, stddev %f" % (timer.interval / nbench, times.std()))
    #print(prof.key_averages().table(sort_by="cuda_time_total"))


if __name__ == "__main__":
    benchmark()
