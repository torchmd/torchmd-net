#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>
using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using std::make_tuple;
using std::max;
using torch::empty;
using torch::full;
using torch::kInt32;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;
using torch::zeros;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <typename scalar_t, int num_dims>
using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x){};
template <> __device__ __forceinline__ float sqrt_(float x) {
    return ::sqrtf(x);
};
template <> __device__ __forceinline__ double sqrt_(double x) {
    return ::sqrt(x);
};

template <typename scalar_t>
__global__ void
forward_kernel(const int32_t num_all_pairs, const Accessor<scalar_t, 2> positions,
               // const Accessor<int32_t, 1> batch_offsets, const int32_t batch_index,
               const Accessor<int32_t, 1> batch, const scalar_t cutoff2,
               Accessor<int32_t, 1> i_curr_pair, Accessor<int32_t, 2> neighbors,
               Accessor<scalar_t, 2> deltas, Accessor<scalar_t, 1> distances) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_all_pairs)
        return;

    int32_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index)
        row--;
    const int32_t column = (index - row * (row - 1) / 2);
    if (batch[row] != batch[column])
        return;
    scalar_t delta_x = positions[row][0] - positions[column][0];
    scalar_t delta_y = positions[row][1] - positions[column][1];
    scalar_t delta_z = positions[row][2] - positions[column][2];
    const scalar_t distance2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    if (distance2 > cutoff2)
        return;

    const int32_t i_pair = atomicAdd(&i_curr_pair[0], 1);
    // We handle too many neighbors outside of the kernel
    if (i_pair < neighbors.size(1)) {
        neighbors[0][i_pair] = row;
        neighbors[1][i_pair] = column;
        deltas[i_pair][0] = delta_x;
        deltas[i_pair][1] = delta_y;
        deltas[i_pair][2] = delta_z;
        distances[i_pair] = sqrt_(distance2);
    }
}

template <typename scalar_t>
__global__ void
backward_kernel(const Accessor<int32_t, 2> neighbors, const Accessor<scalar_t, 2> deltas,
                const Accessor<scalar_t, 1> distances, const Accessor<scalar_t, 1> grad_distances,
                Accessor<scalar_t, 2> grad_positions) {
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs)
        return;

    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    if (i_atom < 0)
        return;

    const int32_t i_comp = blockIdx.z;
    const scalar_t grad = deltas[i_pair][i_comp] / distances[i_pair] * grad_distances[i_pair];
    atomicAdd(&grad_positions[i_atom][i_comp], (i_dir ? -1 : 1) * grad);
}

static void checkInput(const Tensor& positions, const Tensor& batch) {
    // This version works with batches
    // Batch contains the molecule index for each atom in positions
    // Neighbors are only calculated within the same molecule
    // Batch is a 1D tensor of size (N_atoms)
    // Batch is assumed to be sorted and starts at zero.
    // Batch is assumed to be contiguous
    // Batch is assumed to be of type torch::kLong
    // Batch is assumed to be non-negative
    // Each batch can have a different number of atoms
    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0,
                "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    TORCH_CHECK(batch.dim() == 1, "Expected \"batch\" to have one dimension");
    TORCH_CHECK(batch.size(0) == positions.size(0),
                "Expected the 1st dimension size of \"batch\" to be the same as the 1st dimension "
                "size of \"positions\"");
    TORCH_CHECK(batch.is_contiguous(), "Expected \"batch\" to be contiguous");
    TORCH_CHECK(batch.dtype() == torch::kInt32, "Expected \"batch\" to have torch::kInt32 dtype");
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& positions, const Tensor& batch,
                               const Scalar& cutoff, const Scalar& max_num_pairs,
                               bool checkErrors) {
        checkInput(positions, batch);
        const auto max_num_pairs_ = max_num_pairs.toLong();
        TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
        const int num_atoms = positions.size(0);
        const int num_pairs = max_num_pairs_;
        const TensorOptions options = positions.options();
        const auto stream = getCurrentCUDAStream(positions.get_device());
        const Tensor neighbors = full({2, num_pairs}, -1, options.dtype(kInt32));
        const Tensor deltas = empty({num_pairs, 3}, options);
        const Tensor distances = full(num_pairs, 0, options);
        const Tensor i_curr_pair = zeros(1, options.dtype(kInt32));
        {
            const CUDAStreamGuard guard(stream);
            const int num_atoms = positions.size(0);
            const int num_all_pairs = num_atoms * (num_atoms - 1) / 2;
            const int num_threads = 128;
            const int num_blocks = max((num_all_pairs + num_threads - 1) / num_threads, 1);
            AT_DISPATCH_FLOATING_TYPES(
                positions.scalar_type(), "get_neighbor_pairs_forward", [&]() {
                    const scalar_t cutoff_ = cutoff.to<scalar_t>();
                    TORCH_CHECK(cutoff_ > 0, "Expected \"cutoff\" to be positive");
                    forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                        num_all_pairs, get_accessor<scalar_t, 2>(positions),
                        get_accessor<int32_t, 1>(batch), cutoff_ * cutoff_,
                        get_accessor<int32_t, 1>(i_curr_pair), get_accessor<int32_t, 2>(neighbors),
                        get_accessor<scalar_t, 2>(deltas), get_accessor<scalar_t, 1>(distances));
                });
        }
        // Synchronize and check the number of pairs found. Note that this is incompatible with CUDA
        // graphs
        if (checkErrors) {
            int num_found_pairs = i_curr_pair.item<int32_t>();
            TORCH_CHECK(num_found_pairs <= max_num_pairs_,
                        "Too many neighbor pairs found. Maximum is " +
                            std::to_string(max_num_pairs_),
                        " but found " + std::to_string(num_found_pairs));
        }
        neighbors.resize_({2, i_curr_pair[0].item<int>()});
        deltas.resize_({i_curr_pair[0].item<int>(), 3});
        distances.resize_(i_curr_pair[0].item<int>());
        ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;
        return {neighbors, deltas, distances};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {
        const Tensor grad_distances = grad_inputs[1];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks_x = max((num_pairs + num_threads - 1) / num_threads, 1);
        const dim3 blocks(num_blocks_x, 2, 3);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        const tensor_list data = ctx->get_saved_variables();
        const Tensor neighbors = data[0];
        const Tensor deltas = data[1];
        const Tensor distances = data[2];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(
            grad_distances.scalar_type(), "get_neighbor_pairs_backward", [&]() {
                const CUDAStreamGuard guard(stream);
                backward_kernel<<<blocks, num_threads, 0, stream>>>(
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 1>(distances), get_accessor<scalar_t, 1>(grad_distances),
                    get_accessor<scalar_t, 2>(grad_positions));
            });

        return {grad_positions, Tensor(), Tensor()};
    }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_size,
              const Scalar& cutoff, const Scalar& max_num_pairs, bool checkErrors) {
               const tensor_list results =
                   Autograd::apply(positions, batch, cutoff, max_num_pairs, checkErrors);
               return std::make_tuple(results[0], results[1], results[2]);
           });
}
