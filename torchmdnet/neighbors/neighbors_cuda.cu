#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

namespace torch {
static const double radius = 10;

template <typename scalar_t, int num_dims>
    using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

#define HASH_COORDS_TO_INDEX(x,y,z) ((num_partitions_x - 1) * ((num_partitions_y - 1) * z + y) + x)

template <typename scalar_t> __global__ void forward_kernel(
    const Accessor<scalar_t, 2> positions,
    const int32_t max_hash_size,
    const int32_t num_partitions_x,
    const int32_t num_partitions_y,
    const int32_t num_partitions_z,
    Accessor<int32_t, 2> geometric_hash,
    Accessor<int32_t, 1> geometric_hash_sizes,
    Accessor<int32_t, 1> rows,
    Accessor<int32_t, 1> columns,
    Accessor<scalar_t, 2> deltas,
    Accessor<scalar_t, 1> distances,
    int32_t *total_neighbors
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_atoms = positions.size(0);
    if (index >= num_atoms) return;

    const int32_t coord_x = round(positions[index][0] / radius) + 1;
    const int32_t coord_y = round(positions[index][1] / radius) + 1;
    const int32_t coord_z = round(positions[index][2] / radius) + 1;

    const int32_t hash_index = HASH_COORDS_TO_INDEX(coord_x, coord_y, coord_z);

    const int32_t last_size = atomicAdd(&geometric_hash_sizes[hash_index], 1);
    if (last_size == max_hash_size) return;
    geometric_hash[hash_index][last_size] = index;

    const int32_t max_num_neighbors = distances.size(0);

    __syncthreads();

    // TODO: Minimal neighbouring hashes to check are relative positions:
    // (1,-1,-1), (1,-1,0), (1,-1,1), (1,0,-1), (1,0,0), (1,0,1), (0,-1,-1),
    // (0,-1,0), (0,-1,1), (0,0,-1), (0,0,0), (1,-1,1), (1,-1,0), (1,-1,1)

    for (int32_t iter_i = -1; iter_i <= 1; iter_i++) {
        for (int32_t iter_j = -1; iter_j <= 1; iter_j++) {
            for (int32_t iter_k = -1; iter_k <= 1; iter_k++) {
                const int32_t neighbor_hash_index = HASH_COORDS_TO_INDEX(coord_x + iter_i, coord_y + iter_j, coord_z + iter_k);

                for (int iter = 0; iter < geometric_hash_sizes[neighbor_hash_index]; iter++) {
                    const scalar_t delta_x = positions[index][0] - positions[geometric_hash[neighbor_hash_index][iter]][0];
                    const scalar_t delta_y = positions[index][1] - positions[geometric_hash[neighbor_hash_index][iter]][1];
                    const scalar_t delta_z = positions[index][2] - positions[geometric_hash[neighbor_hash_index][iter]][2];

                    const scalar_t distance = sqrt_((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
                    if (distance < radius) {
                        int32_t last_size = atomicAdd(total_neighbors, 1);
                        if (*total_neighbors >= max_num_neighbors) return;

                        columns[last_size] = geometric_hash[hash_index][last_size];
                        rows[last_size] = geometric_hash[neighbor_hash_index][iter];
                        deltas[last_size][0] = delta_x;
                        deltas[last_size][1] = delta_y;
                        deltas[last_size][2] = delta_z;
                        distances[last_size] = distance;
                    }
                }
            }
        }
    }
}

#undef HASH_COORDS_TO_INDEX

template <typename scalar_t> __global__ void backward_kernel(
    const Accessor<int32_t, 1> rows,
    const Accessor<int32_t, 1> columns,
    const Accessor<scalar_t, 2> deltas,
    const Accessor<scalar_t, 1> distances,
    const Accessor<scalar_t, 1> grad_distances,
    Accessor<scalar_t, 2> grad_positions
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_neighbors = distances.size(0);
    if (index >= num_neighbors) return;

    const scalar_t grad = grad_distances[index] / distances[index];
    const scalar_t grad_x = deltas[index][0] * grad;
    const scalar_t grad_y = deltas[index][1] * grad;
    const scalar_t grad_z = deltas[index][2] * grad;

    const int32_t row = rows[index];
    atomicAdd(&grad_positions[row][0], grad_x);
    atomicAdd(&grad_positions[row][1], grad_y);
    atomicAdd(&grad_positions[row][2], grad_z);

    const int32_t column = columns[index];
    atomicAdd(&grad_positions[column][0], -grad_x);
    atomicAdd(&grad_positions[column][1], -grad_y);
    atomicAdd(&grad_positions[column][2], -grad_z);
}

class Autograd : public autograd::Function<Autograd> {
public:
    static autograd::tensor_list forward(autograd::AutogradContext *ctx, const Tensor positions) {

        TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
        TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
        TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

        const int num_atoms = positions.size(0);
        const int num_neighbors = num_atoms * (num_atoms - 1) / 2;
        const int num_threads = 128;
        const int num_blocks = (num_neighbors + num_threads - 1) / num_threads;
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());

        const TensorOptions options = positions.options();
        const Tensor indices = arange(0, num_neighbors, options.dtype(kInt32));

        const Tensor vectors = index_select(positions, 0, indices);

        Tensor max_coords = zeros({3}, kFloat64);

        for(unsigned iter = 0; iter < num_atoms; iter++) {
            const Tensor vector = vectors[iter];

            max_coords[0] = max(vector[0], max_coords[0]);
            max_coords[1] = max(vector[1], max_coords[1]);
            max_coords[2] = max(vector[2], max_coords[2]);
        }

        const int num_partitions_x = ceil((max_coords[0]).item<double>() / radius) + 2;
        const int num_partitions_y = ceil((max_coords[1]).item<double>() / radius) + 2;
        const int num_partitions_z = ceil((max_coords[2]).item<double>() / radius) + 2;
        const int num_partitions = num_partitions_x * num_partitions_y * num_partitions_z;
        const int max_hash_size = 500;

        Tensor geometric_hash = empty({num_partitions, max_hash_size}, options.dtype(kInt32));
        Tensor geometric_hash_sizes = zeros({num_partitions}, options.dtype(kInt32));
        int32_t total_neighbors = 0;

        const Tensor rows = empty(num_neighbors, options.dtype(kInt32));
        const Tensor columns = empty(num_neighbors, options.dtype(kInt32));
        const Tensor deltas = empty({num_neighbors, 3}, options);
        const Tensor distances = empty(num_neighbors, options);

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                max_hash_size,
                num_partitions_x,
                num_partitions_y,
                num_partitions_z,
                get_accessor<int32_t, 2>(geometric_hash),
                get_accessor<int32_t, 1>(geometric_hash_sizes),
                get_accessor<int32_t, 1>(rows),
                get_accessor<int32_t, 1>(columns),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                &total_neighbors);
        });

        ctx->save_for_backward({rows, columns, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;

        return {rows, columns, distances};
    }

    static autograd::tensor_list backward(autograd::AutogradContext* ctx, autograd::tensor_list grad_inputs) {

        const Tensor grad_distances = grad_inputs[2];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_neighbors = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks = (num_neighbors + num_threads - 1) / num_threads;
        const auto stream = c10::cuda::getCurrentCUDAStream(grad_distances.get_device());

        const autograd::tensor_list neighbors = ctx->get_saved_variables();
        const Tensor rows = neighbors[0];
        const Tensor columns = neighbors[1];
        const Tensor deltas = neighbors[2];
        const Tensor distances = neighbors[3];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(grad_distances.scalar_type(), "get_neighbor_list", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            backward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<int32_t, 1>(rows),
                get_accessor<int32_t, 1>(columns),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<scalar_t, 1>(grad_distances),
                get_accessor<scalar_t, 2>(grad_positions));
        });

        return {grad_positions};
    }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_list", [](const Tensor& positions){
        const autograd::tensor_list neighbors = Autograd::apply(positions);
        return std::make_tuple(neighbors[0], neighbors[1], neighbors[2]);
    });
}

}
