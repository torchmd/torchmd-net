#!/bin/bash

set -e
set -x

# Create pip directory if it doesn't exist
mkdir -p "C:\ProgramData\pip"

# Create pip.ini file with PyTorch CPU index
echo "[global]
extra-index-url = https://download.pytorch.org/whl/cpu" > "C:\ProgramData\pip\pip.ini"

if [ "$ACCELERATOR" == "cu118" ]; then
    curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe
    ./cuda.exe -s nvcc_11.8 cudart_11.8 cublas_dev_11.8 curand_dev_11.8 cusparse_dev_11.8 cusolver_dev_11.8 thrust_11.8
    rm cuda.exe

    # Create pip.ini file with PyTorch CUDA 11.8 index
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu118" > "C:\ProgramData\pip\pip.ini"
elif [ "$ACCELERATOR" == "cu126" ]; then
    curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe
    ./cuda.exe -s nvcc_12.6 cudart_12.6 cublas_dev_12.6 curand_dev_12.6 cusparse_dev_12.6 cusolver_dev_12.6 thrust_12.6
    rm cuda.exe

    # Create pip.ini file with PyTorch CUDA 12.6 index
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu126" > "C:\ProgramData\pip\pip.ini"
elif [ "$ACCELERATOR" == "cu128" ]; then
    curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
    ./cuda.exe -s nvcc_12.8 cudart_12.8 cublas_dev_12.8 curand_dev_12.8 cusparse_dev_12.8 cusolver_dev_12.8 thrust_12.8
    rm cuda.exe

    # Create pip.ini file with PyTorch CUDA 12.8 index
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu128" > "C:\ProgramData\pip\pip.ini"
fi