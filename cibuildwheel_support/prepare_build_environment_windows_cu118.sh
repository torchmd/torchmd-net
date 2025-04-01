#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe
./cuda.exe -s nvcc_11.8 cudart_11.8 cublas_dev_11.8 curand_dev_11.8 cusparse_dev_11.8 cusolver_dev_11.8 thrust_11.8
rm cuda.exe

# Create pip directory if it doesn't exist
mkdir -p "C:\ProgramData\pip"

# Create pip.ini file with PyTorch CUDA 11.8 index
echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu118" > "C:\ProgramData\pip\pip.ini"