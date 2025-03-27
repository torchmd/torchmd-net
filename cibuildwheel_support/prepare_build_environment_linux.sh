#! /bin/bash

set -e
set -x


if [ "$CIBW_ARCHS" != "aarch64" ]; then

    # Install CUDA 12.4:
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

    dnf install --setopt=obsoletes=0 -y \
        cuda-nvcc-12-4-12.4.131-1 \
        cuda-cudart-devel-12-4-12.4.127-1 \
        libcurand-devel-12-4-10.3.5.147-1 \
        libcudnn9-devel-cuda-12-9.1.0.70-1 \
        libcublas-devel-12-4-12.4.5.8-1 \
        libnccl-devel-2.21.5-1+cuda12.4 \
        libcusparse-devel-12-4-12.3.1.170-1 \
        libcusolver-devel-12-4-11.6.1.9-1 \
        gcc-toolset-13

    ln -s cuda-12.4 /usr/local/cuda
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/gcc /usr/local/cuda/bin/gcc
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/g++ /usr/local/cuda/bin/g++

    export CUDA_HOME="/usr/local/cuda"
    export WITH_CUDA=1
    
fi

