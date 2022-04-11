#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <algorithm>
#include <tuple>

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using std::make_tuple;
using std::max;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;
using torch::empty;
using torch::full;
using torch::kInt32;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;
using torch::zeros;

template <typename scalar_t, int num_dims>
    using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims> 
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

template <typename scalar_t> __global__ void forward_kernel(
    const Accessor<scalar_t, 2> positions,
    const int32_t num_all_pairs,
    const scalar_t cutoff2,
    Accessor<int32_t, 2> neighbors,
    Accessor<scalar_t, 2> deltas,
    Accessor<scalar_t, 1> distances
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_all_pairs) return;

    int32_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index) row--;
    const int32_t column = index - row * (row - 1) / 2;

    const scalar_t delta_x = positions[row][0] - positions[column][0];
    const scalar_t delta_y = positions[row][1] - positions[column][1];
    const scalar_t delta_z = positions[row][2] - positions[column][2];
    const scalar_t distance2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    if (distance2 > cutoff2) return;

    neighbors[0][index] = row;
    neighbors[1][index] = column;
    deltas[index][0] = delta_x;
    deltas[index][1] = delta_y;
    deltas[index][2] = delta_z;
    distances[index] = sqrt_(distance2);
}

template <typename scalar_t> __global__ void backward_kernel(
    const Accessor<int32_t, 2> neighbors,
    const Accessor<scalar_t, 2> deltas,
    const Accessor<scalar_t, 1> distances,
    const Accessor<scalar_t, 1> grad_distances,
    Accessor<scalar_t, 2> grad_positions
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_neighbors = distances.size(0);
    if (index >= num_neighbors) return;

    const int32_t row = neighbors[0][index];
    if (row < 0) return;

    const int32_t column = neighbors[1][index];
    if (column < 0) return;

    const scalar_t grad = grad_distances[index] / distances[index];
    const scalar_t grad_x = deltas[index][0] * grad;
    const scalar_t grad_y = deltas[index][1] * grad;
    const scalar_t grad_z = deltas[index][2] * grad;

    atomicAdd(&grad_positions[row][0], grad_x);
    atomicAdd(&grad_positions[row][1], grad_y);
    atomicAdd(&grad_positions[row][2], grad_z);

    atomicAdd(&grad_positions[column][0], -grad_x);
    atomicAdd(&grad_positions[column][1], -grad_y);
    atomicAdd(&grad_positions[column][2], -grad_z);
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx,
                               const Tensor& positions,
                               const Scalar& cutoff,
                               const Scalar& max_num_neighbors) {

        TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
        TORCH_CHECK(positions.size(0) > 0, "Expected the 1nd dimension size of \"positions\" to be more than 0");
        TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
        TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

        const int max_num_neighbors_ = max_num_neighbors.to<int>();
        TORCH_CHECK(max_num_neighbors_ > 0, "Expected \"max_num_neighbors\" to be positive");

        // Decide the algorithm
        const int num_atoms = positions.size(0);
        const int num_all_pairs = num_atoms * (num_atoms - 1) / 2;
        const int num_exp_pairs = num_atoms * max_num_neighbors_;
        const bool all_pairs = num_all_pairs <= num_exp_pairs;
        const int num_pairs = all_pairs ? num_all_pairs : num_exp_pairs;

        const int num_threads = 128;
        const int num_blocks = max((num_all_pairs + num_threads - 1) / num_threads, 1);
        const auto stream = getCurrentCUDAStream(positions.get_device());

        const TensorOptions options = positions.options();
        const Tensor neighbors = full({2, num_pairs}, -1, options.dtype(kInt32));
        const Tensor deltas = empty({num_pairs, 3}, options);
        const Tensor distances = full(num_pairs, 0, options);

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_pairs_forward", [&]() {
            const CUDAStreamGuard guard(stream);
            const scalar_t cutoff_ = cutoff.to<scalar_t>();
            TORCH_CHECK(cutoff_ > 0, "Expected \"cutoff\" to be positive");
            forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                num_all_pairs,
                cutoff_ * cutoff_,
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances));
        });

        ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;

        return {neighbors, distances};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {

        const Tensor grad_distances = grad_inputs[1];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks = max((num_pairs + num_threads - 1) / num_threads, 1);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        const tensor_list data = ctx->get_saved_variables();
        const Tensor neighbors = data[0];
        const Tensor deltas = data[1];
        const Tensor distances = data[2];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(grad_distances.scalar_type(), "get_neighbor_pairs_backward", [&]() {
            const CUDAStreamGuard guard(stream);
            backward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<scalar_t, 1>(grad_distances),
                get_accessor<scalar_t, 2>(grad_positions));
        });

        return {grad_positions, Tensor(), Tensor()};
      }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs",
        [](const Tensor& positions, const Scalar& cutoff, const Scalar& max_num_neighbors){
            const tensor_list results = Autograd::apply(positions, cutoff, max_num_neighbors);
            return make_tuple(results[0], results[1]);
    });
}