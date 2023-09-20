import subprocess
from setuptools import setup, find_packages

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, CppExtension
import torch

neighs = CppExtension(
    name='torchmdnet.neighbors.torchmdnet_neighbors',
    sources=["torchmdnet/neighbors/neighbors.cpp", "torchmdnet/neighbors/neighbors_cpu.cpp"],
    include_dirs=include_paths(),
    language='c++')

if torch.cuda._is_compiled():
    neighs = CUDAExtension(
        name='torchmdnet.neighbors.torchmdnet_neighbors',
        sources=["torchmdnet/neighbors/neighbors.cpp", "torchmdnet/neighbors/neighbors_cpu.cpp", "torchmdnet/neighbors/neighbors_cuda.cu"],
        include_dirs=include_paths(),
        language='cuda'
        )

setup(
    name="torchmd-net",
    version=version,
    packages=find_packages(),
    ext_modules=[neighs,],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)},
    include_package_data=True,
    entry_points={"console_scripts": ["torchmd-train = torchmdnet.scripts.train:main"]},
    package_data={"torchmdnet": ["neighbors/torchmdnet_neighbors.so"]},

)
