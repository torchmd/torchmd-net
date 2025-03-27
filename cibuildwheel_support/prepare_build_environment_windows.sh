#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe
./cuda.exe -s nvcc_12.2 cudart_12.2 cublas_dev_12.2 curand_dev_12.2
rm cuda.exe

CUDNN_ROOT="C:/Program Files/NVIDIA/CUDNN/v9.1"
curl --netrc-optional -L -nv -o cudnn.exe https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn_9.1.0_windows.exe
./cudnn.exe -s
sleep 10
# Remove 11.8 folders
rm -rf "$CUDNN_ROOT/bin/11.8"
rm -rf "$CUDNN_ROOT/lib/11.8"
rm -rf "$CUDNN_ROOT/include/11.8"

# Move contents of 12.4 to parent directories
mv "$CUDNN_ROOT/bin/12.4/"* "$CUDNN_ROOT/bin/"
mv "$CUDNN_ROOT/lib/12.4/"* "$CUDNN_ROOT/lib/"
mv "$CUDNN_ROOT/include/12.4/"* "$CUDNN_ROOT/include/"

# Remove empty 12.4 folders
rmdir "$CUDNN_ROOT/bin/12.4"
rmdir "$CUDNN_ROOT/lib/12.4"
rmdir "$CUDNN_ROOT/include/12.4"
cp -r "$CUDNN_ROOT"/* "$CUDA_ROOT"
rm cudnn.exe
