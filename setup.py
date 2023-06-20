import subprocess
from setuptools import setup, find_packages
import os
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"

src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "torchmdnet", "neighbors")
sources = ["neighbors.cpp", "neighbors_cpu.cpp"] + (
    ["neighbors_cuda.cu", "backwards.cu"] if torch.cuda.is_available() else []
)
sources = [os.path.join(src_dir, name) for name in sources]
Ext = CUDAExtension if torch.cuda.is_available() else CppExtension
extension = Ext(
    name="torchmdnet_neighbors", sources=sources
)

setup(
    name="torchmd-net",
    version=version,
    description="TorchMD-Net: A PyTorch-based Deep Learning Framework for Molecular Dynamics",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    entry_points={"console_scripts": ["torchmd-train = torchmdnet.scripts.train:main"]},
)
