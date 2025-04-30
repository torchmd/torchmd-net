#! /bin/bash

set -e
set -x

# Configure pip to use PyTorch extra-index-url for CPU
mkdir -p $HOME/.config/pip
echo "[global]
extra-index-url = https://download.pytorch.org/whl/cpu" > $HOME/.config/pip/pip.conf


if [ "$ACCELERATOR" == "cu118" ]; then

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
    
    # Configure pip to use PyTorch extra-index-url for CUDA 11.8
    mkdir -p $HOME/.config/pip
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu118" > $HOME/.config/pip/pip.conf
    
elif [ "$ACCELERATOR" == "cu126" ]; then
    # Install CUDA 12.6
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

    dnf install --setopt=obsoletes=0 -y \
        cuda-compiler-12-6-12.6.3-1 \
        cuda-libraries-12-6-12.6.3-1 \
        cuda-libraries-devel-12-6-12.6.3-1 \
        cuda-toolkit-12-6-12.6.3-1 \
        gcc-toolset-13

    ln -s cuda-12.6 /usr/local/cuda
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/gcc /usr/local/cuda/bin/gcc
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/g++ /usr/local/cuda/bin/g++
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/libcuda.so.1

    # Configure pip to use PyTorch extra-index-url for CUDA 12.6
    mkdir -p $HOME/.config/pip
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu126" > $HOME/.config/pip/pip.conf

elif [ "$ACCELERATOR" == "cu128" ]; then
    # Install CUDA 12.8
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

    dnf install --setopt=obsoletes=0 -y \
        cuda-compiler-12-8-12.8.1-1 \
        cuda-libraries-12-8-12.8.1-1 \
        cuda-libraries-devel-12-8-12.8.1-1 \
        cuda-toolkit-12-8-12.8.1-1 \
        gcc-toolset-13

    ln -s cuda-12.8 /usr/local/cuda
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/gcc /usr/local/cuda/bin/gcc
    ln -s /opt/rh/gcc-toolset-13/root/usr/bin/g++ /usr/local/cuda/bin/g++
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/libcuda.so.1

    # Configure pip to use PyTorch extra-index-url for CUDA 12.8
    mkdir -p $HOME/.config/pip
    echo "[global]
extra-index-url = https://download.pytorch.org/whl/cu128" > $HOME/.config/pip/pip.conf

fi