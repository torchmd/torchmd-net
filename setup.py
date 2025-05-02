# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import platform


if os.environ.get("ACCELERATOR", None) is not None:
    use_cuda = os.environ.get("ACCELERATOR", "").startswith("cu")
else:
    use_cuda = torch.cuda._is_compiled()


def set_torch_cuda_arch_list():
    """Set the CUDA arch list according to the architectures the current torch installation was compiled for.
    This function is a no-op if the environment variable TORCH_CUDA_ARCH_LIST is already set or if torch was not compiled with CUDA support.
    """
    if use_cuda and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        arch_flags = torch._C._cuda_getArchFlags()
        sm_versions = [x[3:] for x in arch_flags.split() if x.startswith("sm_")]
        formatted_versions = ";".join([f"{y[:-1]}.{y[-1]}" for y in sm_versions])
        formatted_versions += "+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = formatted_versions


set_torch_cuda_arch_list()

extension_root = os.path.join("torchmdnet", "extensions")
neighbor_sources = ["neighbors_cpu.cpp"]
if use_cuda:
    neighbor_sources.append("neighbors_cuda.cu")
neighbor_sources = [
    os.path.join(extension_root, "neighbors", source) for source in neighbor_sources
]

runtime_library_dirs = None
if platform.system() == "Darwin":
    runtime_library_dirs = [
        "@loader_path/../../torch/lib",
        "@loader_path/../../nvidia/cuda_runtime/lib",
    ]
elif platform.system() == "Linux":
    runtime_library_dirs = [
        "$ORIGIN/../../torch/lib",
        "$ORIGIN/../../nvidia/cuda_runtime/lib",
    ]

extra_deps = []
if os.getenv("ACCELERATOR", "").startswith("cu"):
    cuda_ver = os.getenv("ACCELERATOR")[2:4]
    extra_deps = [f"nvidia-cuda-runtime-cu{cuda_ver}"]

ExtensionType = CppExtension if not use_cuda else CUDAExtension
extensions = ExtensionType(
    name="torchmdnet.extensions.torchmdnet_extensions",
    sources=[os.path.join(extension_root, "torchmdnet_extensions.cpp")]
    + neighbor_sources,
    define_macros=[("WITH_CUDA", 1)] if use_cuda else [],
    runtime_library_dirs=runtime_library_dirs,
)

if __name__ == "__main__":
    setup(
        ext_modules=[extensions],
        cmdclass={
            "build_ext": BuildExtension.with_options(
                no_python_abi_suffix=True, use_ninja=False
            )
        },
        install_requires=[
            "h5py",
            # "nnpops",
            "torch>=2.5",
            "torch_geometric",
            "lightning",
            "tqdm",
            "numpy",
        ]
        + extra_deps,
    )
