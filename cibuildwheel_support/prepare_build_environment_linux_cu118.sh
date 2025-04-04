#! /bin/bash

set -e
set -x


if [ "$CIBW_ARCHS" != "aarch64" ]; then

    # Install CUDA 11.8:
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

    dnf install --setopt=obsoletes=0 -y \
        cuda-nvcc-11-8-11.8.89-1 \
        cuda-cudart-devel-11-8-11.8.89-1 \
        libcurand-devel-11-8-10.3.0.86-1 \
        libcudnn9-devel-cuda-11-9.8.0.87-1 \
        libcublas-devel-11-8-11.11.3.6-1 \
        libnccl-devel-2.15.5-1+cuda11.8 \
        libcusparse-devel-11-8-11.7.5.86-1 \
        libcusolver-devel-11-8-11.4.1.48-1 \
        gcc-toolset-11

    ln -s cuda-11.8 /usr/local/cuda
    ln -s /opt/rh/gcc-toolset-11/root/usr/bin/gcc /usr/local/cuda/bin/gcc
    ln -s /opt/rh/gcc-toolset-11/root/usr/bin/g++ /usr/local/cuda/bin/g++

    export CUDA_HOME="/usr/local/cuda"
    export WITH_CUDA=1

    
    # Configure pip to use PyTorch extra-index-url for CUDA 11.8
    mkdir -p $HOME/.config/pip
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu118" > $HOME/.config/pip/pip.conf
    
fi

