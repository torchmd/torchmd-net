/* Raul P. Pelaez 2023. Torch extensions to the torchmdnet library.
 * You can expose functions to python here which will be compatible with TorchScript.
 * Add your exports to the TORCH_LIBRARY macro below, see __init__.py to see how to access them from python.
 */


#include <torch/extension.h>
#if defined(WITH_CUDA)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

/* @brief Returns true if the current torch CUDA stream is capturing.
 * This function is required because the one available in torch is not compatible with TorchScript.
 * @return True if the current torch CUDA stream is capturing.
 */
bool is_current_stream_capturing() {
#if defined(WITH_CUDA)
  auto current_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus capture_status;
  cudaError_t err = cudaStreamGetCaptureInfo(current_stream, &capture_status, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  return capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}


TORCH_LIBRARY(torchmdnet_extensions, m) {
    m.def("is_current_stream_capturing", is_current_stream_capturing);
}
