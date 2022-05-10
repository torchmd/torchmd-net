#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

static const int num_threads = 128;

template <typename scalar_t, int num_dims>
    using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dims, torch::RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const torch::Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

template <typename scalar_t> __global__ void kernel_get_maxmin_coords(
    const Accessor<scalar_t, 2> positions,
    Accessor<scalar_t, 2> result
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_atoms = positions.size(0);

    __shared__ scalar_t buffer[2][num_threads][3];

    if (index > num_atoms) return;

    const int16_t maxmin = blockIdx.z;
    const int16_t coord = blockIdx.y;

    if (maxmin == 0) {
        buffer[maxmin][threadIdx.x][coord] = positions[index][coord];

        __syncthreads();

        for (int16_t stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                if (buffer[maxmin][threadIdx.x + stride][coord] > buffer[maxmin][threadIdx.x][coord]) {
                    buffer[maxmin][threadIdx.x][coord] = buffer[maxmin][threadIdx.x + stride][coord];
                }
            }

            __syncthreads();
        }

        if (index == 0) {
            result[maxmin][coord] = buffer[maxmin][0][coord];
        }
    } else {
        buffer[maxmin][threadIdx.x][coord] = positions[index][coord];

        __syncthreads();

        for (int16_t stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                if (buffer[maxmin][threadIdx.x + stride][coord] < buffer[maxmin][threadIdx.x][coord]) {
                    buffer[maxmin][threadIdx.x][coord] = buffer[maxmin][threadIdx.x + stride][coord];
                }
            }

            __syncthreads();
        }

        if (index == 0) {
            result[maxmin][coord] = buffer[maxmin][0][coord];
        }
    }
}

template <typename scalar_t> __global__ void forward_kernel_get_hashes(
    const Accessor<scalar_t, 2> positions,
    const Accessor<scalar_t, 1> boundary,
    const int64_t max_hash_size,
    const double radius,
    Accessor<int32_t, 4> geometric_hash,
    Accessor<int32_t, 3> geometric_hash_sizes
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_atoms = positions.size(0);
    if (index >= num_atoms) return;

    const int32_t coord_x = floor((positions[index][0] - boundary[0]) / radius) + 1;
    const int32_t coord_y = floor((positions[index][1] - boundary[1]) / radius) + 1;
    const int32_t coord_z = floor((positions[index][2] - boundary[2]) / radius) + 1;

    const int32_t last_size = atomicAdd(&geometric_hash_sizes[coord_x][coord_y][coord_z], 1);
    if (last_size >= max_hash_size) {
        geometric_hash_sizes[coord_x][coord_y][coord_z] = max_hash_size;
        return;
    }
    geometric_hash[coord_x][coord_y][coord_z][last_size] = index;
}

template <typename scalar_t> __global__ void forward_kernel_get_neighbors(
    const Accessor<scalar_t, 2> positions,
    const Accessor<scalar_t, 1> boundary,
    const Accessor<int32_t, 4> geometric_hash,
    const Accessor<int32_t, 3> geometric_hash_sizes,
    const double radius,
    const int64_t max_num_neighbors,
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

    int32_t coord_x = floor((positions[index][0] - boundary[0]) / radius) + 1;
    int32_t coord_y = floor((positions[index][1] - boundary[1]) / radius) + 1;
    int32_t coord_z = floor((positions[index][2] - boundary[2]) / radius) + 1;

    const int32_t coord_offsets[14][3] = {
        { 1,  1,  1 },
        { 1,  1,  0 },
        { 1,  1, -1 },
        { 1,  0,  1 },
        { 1,  0,  0 },
        { 1,  0, -1 },
        { 1, -1,  1 },
        { 1, -1,  0 },
        { 1, -1, -1 },
        { 0,  1,  1 },
        { 0,  1,  0 },
        { 0,  1, -1 },
        { 0,  0,  1 },
        { 0,  0,  0 }
    };

    const int subindex = global_index % 14;
    const auto coord_offset = coord_offsets[subindex];

    coord_x += coord_offset[0];
    coord_y += coord_offset[1];
    coord_z += coord_offset[2];

    int iter = 0;
    if (subindex == 13) {
        while(index != geometric_hash[coord_x][coord_y][coord_z][iter]) {
            iter++;
            if(iter >= geometric_hash_sizes[coord_x][coord_y][coord_z]) {
                break;
            }
        }
        iter++;
    }

    const double radius_squared = radius * radius;
    for (; iter < geometric_hash_sizes[coord_x][coord_y][coord_z]; iter++) {
        const scalar_t delta_x = positions[index][0] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][0];
        const scalar_t delta_y = positions[index][1] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][1];
        const scalar_t delta_z = positions[index][2] - positions[geometric_hash[coord_x][coord_y][coord_z][iter]][2];

        const scalar_t distance_squared = (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z);
        if (distance_squared < radius_squared) {
            int32_t last_size = atomicAdd(&neighbors_number[0], 1);
            if (last_size >= max_num_neighbors) {
                neighbors_number[0] = max_num_neighbors;
                return;
            }

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

class Autograd : public torch::autograd::Function<Autograd> {
public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext *ctx, const torch::Tensor positions, const double radius, const int64_t max_num_neighbors) {

        TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
        TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
        TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

        const int32_t num_atoms = positions.size(0);
        const int32_t num_blocks = (num_atoms + num_threads + 1) / num_threads;
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const int32_t max_hash_size = num_atoms;

        const torch::TensorOptions options = positions.options();
        const torch::Tensor indices = torch::arange(0, num_atoms, options.dtype(torch::kInt32));

        const torch::Tensor vectors = torch::index_select(positions, 0, indices);

        const int32_t divisions = ceil(((double)num_atoms) / ((double)num_threads));
        const torch::Tensor boundary = torch::zeros({2, 3}, options.dtype(torch::kFloat32));

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_maxmin_coords", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            const dim3 num_blocks(divisions, 3, 2);
            kernel_get_maxmin_coords<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                get_accessor<scalar_t, 2>(boundary));
        });

        const int num_partitions_x = ceil((boundary[0][0] - boundary[1][0]).item<double>() / radius) + 2;
        const int num_partitions_y = ceil((boundary[0][1] - boundary[1][1]).item<double>() / radius) + 2;
        const int num_partitions_z = ceil((boundary[0][2] - boundary[1][2]).item<double>() / radius) + 2;

        torch::Tensor geometric_hash = torch::zeros({num_partitions_x, num_partitions_y, num_partitions_z, max_hash_size}, options.dtype(torch::kInt32));
        torch::Tensor geometric_hash_sizes = torch::zeros({num_partitions_x, num_partitions_y, num_partitions_z}, options.dtype(torch::kInt32));

        const torch::Tensor rows = torch::full(max_num_neighbors, -1, options.dtype(torch::kInt32));
        const torch::Tensor columns = torch::full(max_num_neighbors, -1, options.dtype(torch::kInt32));
        const torch::Tensor deltas = torch::empty({max_num_neighbors, 3}, options);
        const torch::Tensor distances = torch::empty(max_num_neighbors, options);

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list_step1", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            forward_kernel_get_hashes<<<num_blocks, num_threads, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                get_accessor<scalar_t, 1>(boundary[1]),
                max_hash_size,
                radius,
                get_accessor<int32_t, 4>(geometric_hash),
                get_accessor<int32_t, 3>(geometric_hash_sizes));
        });

        const size_t num_blocks_2 = num_blocks * 2;
        const size_t num_threads_2 = num_threads * 7;


        const torch::Tensor neighbors_number = torch::zeros(1, options.dtype(torch::kInt32));

        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_list_step2", [&]() {
            const c10::cuda::CUDAStreamGuard guard(stream);
            forward_kernel_get_neighbors<<<num_blocks_2, num_threads_2, 0, stream>>>(
                get_accessor<scalar_t, 2>(positions),
                get_accessor<scalar_t, 1>(boundary[1]),
                get_accessor<int32_t, 4>(geometric_hash),
                get_accessor<int32_t, 3>(geometric_hash_sizes),
                radius,
                max_num_neighbors,
                get_accessor<int32_t, 1>(rows),
                get_accessor<int32_t, 1>(columns),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<int32_t, 1>(neighbors_number));
        });

        const torch::Tensor result_indices = torch::arange(0, neighbors_number.item<int32_t>(), options.dtype(torch::kInt32));

        const torch::Tensor result_rows = torch::index_select(rows, 0, result_indices);
        const torch::Tensor result_columns = torch::index_select(columns, 0, result_indices);
        const torch::Tensor result_deltas = torch::index_select(deltas, 0, result_indices);
        const torch::Tensor result_distances = torch::index_select(distances, 0, result_indices);

        ctx->save_for_backward({result_rows, result_columns, result_deltas, result_distances});
        ctx->saved_data["num_atoms"] = num_atoms;

        return {result_rows, result_columns, result_distances};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_inputs) {

        const torch::Tensor grad_distances = grad_inputs[2];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_neighbors = grad_distances.size(0);
        const int num_blocks = (num_neighbors + num_threads - 1) / num_threads;
        const auto stream = c10::cuda::getCurrentCUDAStream(grad_distances.get_device());

        const torch::autograd::tensor_list neighbors = ctx->get_saved_variables();
        const torch::Tensor rows = neighbors[0];
        const torch::Tensor columns = neighbors[1];
        const torch::Tensor deltas = neighbors[2];
        const torch::Tensor distances = neighbors[3];
        const torch::Tensor grad_positions = torch::zeros({num_atoms, 3}, grad_distances.options());

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

        return {grad_positions, torch::Tensor(), torch::Tensor()};
    }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_list", [](const torch::Tensor& positions, const double radius, const int64_t max_num_neighbors){
        const torch::autograd::tensor_list neighbors = Autograd::apply(positions, radius, max_num_neighbors);
        return std::make_tuple(neighbors[0], neighbors[1], neighbors[2]);
    });
}
