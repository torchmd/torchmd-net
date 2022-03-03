#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using std::tuple;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;
using torch::Tensor;
using torch::TensorOptions;

template <typename scalar_t, int num_dims>
    using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims> 
inline Accessor<scalar_t, num_dims> get_accessor(Tensor tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

template <typename scalar_t> __global__ void kernel(
    const Accessor<scalar_t, 2> positions,
    Accessor<int32_t, 1> rows,
    Accessor<int32_t, 1> columns,
    Accessor<scalar_t, 1> distances
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_neighbors = distances.size(0);
    if (index >= num_neighbors) return;

    const int32_t num_atoms = positions.size(0);
    const int32_t row = index/(num_atoms - 1);
    const int32_t column = (index + index/num_atoms + 1) % num_atoms;

    const scalar_t delta_x = positions[row][0] - positions[column][0];
    const scalar_t delta_y = positions[row][1] - positions[column][1];
    const scalar_t delta_z = positions[row][2] - positions[column][2];
    const scalar_t distance = sqrt_(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);

    rows[index] = row;
    columns[index] = column;
    distances[index] = distance;
}

static tuple<Tensor, Tensor, Tensor> get_neighbor_list(Tensor positions) {

    TORCH_CHECK(positions.is_contiguous(), "get_neighbor_list: expected \"positions\" to be contiguous");

    const int32_t num_atoms = positions.size(0);
    const int32_t num_neighbors = num_atoms * (num_atoms - 1);
    const auto options = TensorOptions().device(positions.device());

    const int num_threads = 128;
    const int num_blocks = (num_neighbors + num_threads - 1) / num_threads;
    const auto stream = getCurrentCUDAStream(positions.get_device());

    const Tensor rows = torch::empty(num_neighbors, options.dtype(torch::kInt32));
    const Tensor columns = torch::empty(num_neighbors, options.dtype(torch::kInt32));
    const Tensor distances = torch::empty(num_neighbors, options.dtype(positions.dtype()));

    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list", [&]() {
        CUDAStreamGuard guard(stream);
        kernel<<<num_blocks, num_threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(positions),
            get_accessor<int32_t, 1>(rows),
            get_accessor<int32_t, 1>(columns),
            get_accessor<scalar_t, 1>(distances));
    });

    return {rows, columns, distances};
}

TORCH_LIBRARY_IMPL(neighbors, CUDA, m) {
    m.impl("get_neighbor_list", &get_neighbor_list);
}