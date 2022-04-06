#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

namespace torch {
static const double radius = 10;
static const double radius_squared = radius *  radius;

static const int num_threads = 128;

template <typename scalar_t, int num_dims>
    using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

template <typename scalar_t> __global__ void kernel_get_max_coords(
    const Accessor<scalar_t, 2> positions,
    Accessor<scalar_t, 2> result
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_atoms = positions.size(0);

    scalar_t max_x = 0.0;
    scalar_t max_y = 0.0;
    scalar_t max_z = 0.0;

    for (int32_t iter = index; iter < num_atoms; iter += blockDim.x * gridDim.x) {
        if (positions[iter][0] > max_x) {
            max_x = positions[iter][0];
        }
        if (positions[iter][1] > max_y) {
            max_y = positions[iter][1];
        }
        if (positions[iter][2] > max_z) {
            max_z = positions[iter][2];
        }
    }

    __shared__ scalar_t buf[num_threads][3];

    buf[threadIdx.x][0] = max_x;
    buf[threadIdx.x][1] = max_y;
    buf[threadIdx.x][2] = max_z;

    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (buf[threadIdx.x + stride][0] > buf[threadIdx.x][0]) {
                buf[threadIdx.x][0] = buf[threadIdx.x + stride][0];
            }
            if (buf[threadIdx.x + stride][1] > buf[threadIdx.x][1]) {
                buf[threadIdx.x][1] = buf[threadIdx.x + stride][1];
            }
            if (buf[threadIdx.x + stride][2] > buf[threadIdx.x][2]) {
                buf[threadIdx.x][2] = buf[threadIdx.x + stride][2];
            }

            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x][0] = buf[0][0];
        result[blockIdx.x][1] = buf[0][1];
        result[blockIdx.x][2] = buf[0][2];
    }
}

template <typename scalar_t> __global__ void forward_kernel_get_hashes(
    const Accessor<scalar_t, 2> positions,
    const int32_t max_hash_size,
    Accessor<int32_t, 4> geometric_hash,
    Accessor<int32_t, 3> geometric_hash_sizes
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_atoms = positions.size(0);
    if (index >= num_atoms) return;

    const int32_t coord_x = round(positions[index][0] / radius) + 1;
    const int32_t coord_y = round(positions[index][1] / radius) + 1;
    const int32_t coord_z = round(positions[index][2] / radius) + 1;

    const int32_t last_size = atomicAdd(&geometric_hash_sizes[coord_x][coord_y][coord_z], 1);
    if (last_size >= max_hash_size) return;
    geometric_hash[coord_x][coord_y][coord_z][last_size] = index;
}

template <typename scalar_t> __global__ void forward_kernel_get_neighbors(
    const Accessor<scalar_t, 2> positions,
    const Accessor<int32_t, 4> geometric_hash,
    const Accessor<int32_t, 3> geometric_hash_sizes,
    Accessor<int32_t, 1> rows,
    Accessor<int32_t, 1> columns,
    Accessor<scalar_t, 2> deltas,
    Accessor<scalar_t, 1> distances,
    Accessor<int32_t, 1> neighbors_number
) {
    const int32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t index = global_index / 14;

    const int32_t num_atoms = positions.size(0);
    if (index >= num_atoms) return;

    int32_t coord_x = round(positions[index][0] / radius) + 1;
    int32_t coord_y = round(positions[index][1] / radius) + 1;
    int32_t coord_z = round(positions[index][2] / radius) + 1;

    const int32_t max_num_neighbors = distances.size(0);

    int iter = 0;

    switch (global_index % 14) {
        case 0:
            coord_x += 1; coord_y += 1; coord_z += 1;
            break;
        case 1:
            coord_x += 1; coord_y += 1; coord_z += 0;
            break;
        case 2:
            coord_x += 1; coord_y += 1; coord_z += -1;
            break;
        case 3:
            coord_x += 1; coord_y += 0; coord_z += 1;
            break;
        case 4:
            coord_x += 1; coord_y += 0; coord_z += 0;
            break;
        case 5:
            coord_x += 1; coord_y += 0; coord_z += -1;
            break;
        case 6:
            coord_x += 1; coord_y += -1; coord_z += 1;
            break;
        case 7:
            coord_x += 1; coord_y += -1; coord_z += 0;
            break;
        case 8:
            coord_x += 1; coord_y += -1; coord_z += -1;
            break;
        case 9:
            coord_x += 0; coord_y += 1; coord_z += 1;
            break;
        case 10:
            coord_x += 0; coord_y += 1; coord_z += 0;
            break;
        case 11:
            coord_x += 0; coord_y += 1; coord_z += -1;
            break;
        case 12:
            coord_x += 0; coord_y += 0; coord_z += 1;
            break;
        case 13:
            coord_x += 0; coord_y += 0; coord_z += 0;

            while(index != geometric_hash[coord_x][coord_y][coord_z][iter]) {
                iter++;
            }
            iter++;

            break;
    }

    for (; iter < geometric_hash_sizes[coord_x][coord_y][coord_z]; iter++) {
        const scalar_t delta_x = positions[index][0] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][0];
        const scalar_t delta_y = positions[index][1] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][1];
        const scalar_t delta_z = positions[index][2] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][2];

        const scalar_t distance_squared = (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z);
        if (distance_squared < radius_squared) {
            int32_t last_size = atomicAdd(&neighbors_number[0], 1);
            if (neighbors_number[0] >= max_num_neighbors) return;

            columns[last_size] = index;
            rows[last_size] = geometric_hash[coord_x][coord_y][coord_z][iter];
            deltas[last_size][0] = delta_x;
            deltas[last_size][1] = delta_y;
            deltas[last_size][2] = delta_z;
            distances[last_size] = sqrt_(distance_squared);
        }
    }
}

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
        const int num_blocks = (num_atoms + num_threads + 1) / num_threads;
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());

        const TensorOptions options = positions.options();
        const Tensor indices = arange(0, num_atoms, options.dtype(kInt32));

        const Tensor vectors = index_select(positions, 0, indices);

        const int32_t divisions = 50;
        const Tensor max_coords = zeros({divisions, 3}, options.dtype(kFloat32));

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_max_coords_step1", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            kernel_get_max_coords<<<divisions, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                get_accessor<scalar_t, 2>(max_coords));
        });

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_max_coords_step2", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            kernel_get_max_coords<<<1, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(max_coords),
                get_accessor<scalar_t, 2>(max_coords));
        });

        const int num_partitions_x = ceil((max_coords[0][0]).item<double>() / radius) + 2;
        const int num_partitions_y = ceil((max_coords[0][1]).item<double>() / radius) + 2;
        const int num_partitions_z = ceil((max_coords[0][2]).item<double>() / radius) + 2;
        const int max_hash_size = 500;

        Tensor geometric_hash = empty({num_partitions_x, num_partitions_y, num_partitions_z, max_hash_size}, options.dtype(kInt32));
        Tensor geometric_hash_sizes = zeros({num_partitions_x, num_partitions_y, num_partitions_z}, options.dtype(kInt32));

        const Tensor debug = empty({num_atoms, 8}, options.dtype(kInt32));
        const Tensor rows = empty(num_neighbors, options.dtype(kInt32));
        const Tensor columns = empty(num_neighbors, options.dtype(kInt32));
        const Tensor deltas = empty({num_neighbors, 3}, options);
        const Tensor distances = empty(num_neighbors, options);
        const Tensor neighbors_number = empty(1, options.dtype(kInt32));

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list_step1", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            forward_kernel_get_hashes<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                max_hash_size,
                get_accessor<int32_t, 4>(geometric_hash),
                get_accessor<int32_t, 3>(geometric_hash_sizes));
        });

        std::cout << "test1" << std::endl;

        const size_t num_blocks_2 = num_blocks * 2;
        const size_t num_threads_2 = num_threads * 7;

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list_step2", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            forward_kernel_get_neighbors<<<num_blocks_2, num_threads_2, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                get_accessor<int32_t, 4>(geometric_hash),
                get_accessor<int32_t, 3>(geometric_hash_sizes),
                get_accessor<int32_t, 1>(rows),
                get_accessor<int32_t, 1>(columns),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<int32_t, 1>(neighbors_number));
        });

        std::cout << "test2" << std::endl;

        const Tensor result_indices = arange(0, neighbors_number.item<int32_t>(), options.dtype(kInt32));

        const Tensor result_rows = index_select(rows, 0, result_indices);
        const Tensor result_columns = index_select(columns, 0, result_indices);
        const Tensor result_deltas = index_select(deltas, 0, result_indices);
        const Tensor result_distances = index_select(distances, 0, result_indices);

        ctx->save_for_backward({result_rows, result_columns, result_deltas, result_distances});
        ctx->saved_data["num_atoms"] = num_atoms;

        return {result_rows, result_columns, result_distances};
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

        AT_DISPATCH_FLOATING_TYPES(grad_distances.scalar_type(), "backward_get_neighbor_list", [&]() {
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
