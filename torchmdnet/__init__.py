from importlib.metadata import version, PackageNotFoundError
import sys
import os

try:
    __version__ = version("torchmd-net")
except PackageNotFoundError:
    # package is not installed
    pass

python_interpreter = sys.executable
CUDA_HOME = os.path.dirname(os.path.dirname(python_interpreter))
print(f"CUDA_HOME: {CUDA_HOME}")
os.environ["CUDA_HOME"] = CUDA_HOME
