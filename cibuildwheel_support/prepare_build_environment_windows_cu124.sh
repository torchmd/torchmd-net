#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe
./cuda.exe -s nvcc_12.4 cudart_12.4 cublas_dev_12.4 curand_dev_12.4 cusparse_dev_12.4 cusolver_dev_12.4 thrust_12.4 npp_dev_12.4
rm cuda.exe

export WITH_CUDA=1
export CUDA_HOME="$CUDA_ROOT"

# Debug CUDA installation
ls -l "$CUDA_ROOT/bin"
nvcc --version