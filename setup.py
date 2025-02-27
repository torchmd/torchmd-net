# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os


# If WITH_CUDA is defined
env_with_cuda = os.getenv("WITH_CUDA")
if env_with_cuda is not None:
    if env_with_cuda not in ("0", "1"):
        raise ValueError(
            f"WITH_CUDA environment variable got invalid value {env_with_cuda}. Expected '0' or '1'"
        )
    use_cuda = env_with_cuda == "1"
else:
    use_cuda = torch.cuda._is_compiled()


def set_torch_cuda_arch_list():
    """Set the CUDA arch list according to the architectures the current torch installation was compiled for.
    This function is a no-op if the environment variable TORCH_CUDA_ARCH_LIST is already set or if torch was not compiled with CUDA support.
    """
    if use_cuda and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        arch_flags = torch._C._cuda_getArchFlags()
        sm_versions = [x[3:] for x in arch_flags.split() if x.startswith("sm_")]
        formatted_versions = ";".join([f"{y[0]}.{y[1]}" for y in sm_versions])
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

ExtensionType = CppExtension if not use_cuda else CUDAExtension
extensions = ExtensionType(
    name="torchmdnet.extensions.torchmdnet_extensions",
    sources=[os.path.join(extension_root, "torchmdnet_extensions.cpp")]
    + neighbor_sources,
    define_macros=[("WITH_CUDA", 1)] if use_cuda else [],
)

if __name__ == "__main__":
    setup(
        ext_modules=[extensions],
        cmdclass={
            "build_ext": BuildExtension.with_options(
                no_python_abi_suffix=True, use_ninja=False
            )
        },
    )
