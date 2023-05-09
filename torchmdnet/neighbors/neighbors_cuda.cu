#include "common.cuh"
#include <algorithm>
#include <torch/extension.h>

__device__ uint32_t get_row(uint32_t index) {
    uint32_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index)
        row--;
    return row;
}

template <typename scalar_t>
__global__ void forward_kernel_brute(uint32_t num_all_pairs, const Accessor<scalar_t, 2> positions,
                                     const Accessor<int64_t, 1> batch, scalar_t cutoff_lower2,
                                     scalar_t cutoff_upper2, bool loop, bool include_transpose,
                                     Accessor<int32_t, 1> i_curr_pair,
                                     Accessor<int32_t, 2> neighbors, Accessor<scalar_t, 2> deltas,
                                     Accessor<scalar_t, 1> distances, bool use_periodic,
                                     const Accessor<scalar_t, 2> box_vectors) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_all_pairs)
        return;
    const uint32_t row = get_row(index);
    const uint32_t column = (index - row * (row - 1) / 2);
    if (batch[row] == batch[column]) {
        const scalar3<scalar_t> pos_i{positions[row][0], positions[row][1], positions[row][2]};
        const scalar3<scalar_t> pos_j{positions[column][0], positions[column][1],
                                      positions[column][2]};
        const auto delta = triclinic::compute_distance(pos_i, pos_j, use_periodic, box_vectors);
        const scalar_t distance2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (distance2 < cutoff_upper2 && distance2 >= cutoff_lower2) {
            const int32_t i_pair = atomicAdd(&i_curr_pair[0], include_transpose ? 2 : 1);
            // We handle too many neighbors outside of the kernel
            if (i_pair + include_transpose < neighbors.size(1)) {
                const scalar_t r2 = sqrt_(distance2);
                neighbors[0][i_pair] = row;
                neighbors[1][i_pair] = column;
                deltas[i_pair][0] = delta.x;
                deltas[i_pair][1] = delta.y;
                deltas[i_pair][2] = delta.z;
                distances[i_pair] = r2;
                if (include_transpose) {
                    neighbors[0][i_pair + 1] = column;
                    neighbors[1][i_pair + 1] = row;
                    deltas[i_pair + 1][0] = -delta.x;
                    deltas[i_pair + 1][1] = -delta.y;
                    deltas[i_pair + 1][2] = -delta.z;
                    distances[i_pair + 1] = r2;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void add_self_kernel(const int num_atoms, Accessor<scalar_t, 2> positions,
                                Accessor<int32_t, 1> i_curr_pair, Accessor<int32_t, 2> neighbors,
                                Accessor<scalar_t, 2> deltas, Accessor<scalar_t, 1> distances) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    const int32_t i_pair = atomicAdd(&i_curr_pair[0], 1);
    if (i_pair < neighbors.size(1)) {
        neighbors[0][i_pair] = i_atom;
        neighbors[1][i_pair] = i_atom;
        deltas[i_pair][0] = 0;
        deltas[i_pair][1] = 0;
        deltas[i_pair][2] = 0;
        distances[i_pair] = 0;
    }
}

template <int BLOCKSIZE, typename scalar_t>
__global__ void forward_kernel_shared(uint32_t num_atoms, const Accessor<scalar_t, 2> positions,
                                      const Accessor<int64_t, 1> batch, scalar_t cutoff_lower2,
                                      scalar_t cutoff_upper2, bool loop, bool include_transpose,
                                      Accessor<int32_t, 1> i_curr_pair,
                                      Accessor<int32_t, 2> neighbors, Accessor<scalar_t, 2> deltas,
                                      Accessor<scalar_t, 1> distances, int32_t num_tiles,
                                      bool use_periodic, const Accessor<scalar_t, 2> box_vectors) {
    // A thread per atom
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    // All threads must pass through __syncthreads,
    // but when N is not a multiple of 32 some threads are assigned a particle i>N.
    // This threads cant return, so they are masked to not do any work
    const bool active = id < num_atoms;
    __shared__ scalar3<scalar_t> sh_pos[BLOCKSIZE];
    __shared__ int64_t sh_batch[BLOCKSIZE];
    scalar3<scalar_t> pos_i;
    int64_t batch_i;
    if (active) {
        pos_i = {positions[id][0], positions[id][1], positions[id][2]};
        batch_i = batch[id];
    }
    // Distribute the N particles in a group of tiles. Storing in each tile blockDim.x values in
    // shared memory. This way all threads are accesing the same memory addresses at the same time
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load this tiles particles values to shared memory
        const int i_load = tile * blockDim.x + threadIdx.x;
        if (i_load < num_atoms) { // Even if im not active, my thread may load a value each tile to
            // shared memory.
            sh_pos[threadIdx.x] = {positions[i_load][0], positions[i_load][1],
                                   positions[i_load][2]};
            sh_batch[threadIdx.x] = batch[i_load];
        }
        // Wait for all threads to arrive
        __syncthreads();
        // Go through all the particles in the current tile
#pragma unroll 8
        for (int counter = 0; counter < blockDim.x; counter++) {
            if (!active)
                break; // An out of bounds thread must be masked
            const int cur_j = tile * blockDim.x + counter;
            const bool testPair = cur_j < num_atoms and (cur_j < id or (loop and cur_j == id));
            if (testPair) {
                const auto batch_j = sh_batch[counter];
                if (batch_i == batch_j) {
                    const auto pos_j = sh_pos[counter];
                    const auto delta =
                        triclinic::compute_distance(pos_i, pos_j, use_periodic, box_vectors);
                    const scalar_t distance2 =
                        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                    if (distance2 < cutoff_upper2 && distance2 >= cutoff_lower2) {
                        const bool requires_transpose = include_transpose && !(cur_j == id);
                        const int32_t i_pair =
                            atomicAdd(&i_curr_pair[0], requires_transpose ? 2 : 1);
                        if (i_pair + requires_transpose < neighbors.size(1)) {
                            const auto distance = sqrt_(distance2);
                            neighbors[0][i_pair] = id;
                            neighbors[1][i_pair] = cur_j;
                            deltas[i_pair][0] = delta.x;
                            deltas[i_pair][1] = delta.y;
                            deltas[i_pair][2] = delta.z;
                            distances[i_pair] = distance;
                            if (requires_transpose) {
                                neighbors[0][i_pair + 1] = cur_j;
                                neighbors[1][i_pair + 1] = id;
                                deltas[i_pair + 1][0] = -delta.x;
                                deltas[i_pair + 1][1] = -delta.y;
                                deltas[i_pair + 1][2] = -delta.z;
                                distances[i_pair + 1] = distance;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
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
    TORCH_CHECK(positions.size(0) < 1l << 15l,
                "Expected the 1st dimension size of \"positions\" to be less than ", 1l << 15l);
    TORCH_CHECK(batch.dim() == 1, "Expected \"batch\" to have one dimension");
    TORCH_CHECK(batch.size(0) == positions.size(0),
                "Expected the 1st dimension size of \"batch\" to be the same as the 1st dimension "
                "size of \"positions\"");
    TORCH_CHECK(batch.is_contiguous(), "Expected \"batch\" to be contiguous");
    TORCH_CHECK(batch.dtype() == torch::kInt64, "Expected \"batch\" to be of type torch::kLong");
}

enum class strategy { brute, shared };

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& positions, const Tensor& batch,
                               const Scalar& cutoff_lower, const Scalar& cutoff_upper,
                               const Tensor& box_vectors, bool use_periodic,
                               const Scalar& max_num_pairs, bool loop, bool include_transpose,
                               strategy strat) {
        checkInput(positions, batch);
        const auto max_num_pairs_ = max_num_pairs.toLong();
        TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
        if (use_periodic) {
            TORCH_CHECK(box_vectors.dim() == 2, "Expected \"box_vectors\" to have two dimensions");
            TORCH_CHECK(box_vectors.size(0) == 3 && box_vectors.size(1) == 3,
                        "Expected \"box_vectors\" to have shape (3, 3)");
        }
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
            const int32_t num_atoms = positions.size(0);
            if (strat == strategy::brute) {
                const uint64_t num_all_pairs = num_atoms * (num_atoms - 1ul) / 2ul;
                const uint64_t num_threads = 128;
                const uint64_t num_blocks =
                    std::max((num_all_pairs + num_threads - 1ul) / num_threads, 1ul);
                AT_DISPATCH_FLOATING_TYPES(
                    positions.scalar_type(), "get_neighbor_pairs_forward", [&]() {
                        const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
                        const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
                        TORCH_CHECK(cutoff_upper_ > 0, "Expected \"cutoff\" to be positive");
                        forward_kernel_brute<<<num_blocks, num_threads, 0, stream>>>(
                            num_all_pairs, get_accessor<scalar_t, 2>(positions),
                            get_accessor<int64_t, 1>(batch), cutoff_lower_ * cutoff_lower_,
                            cutoff_upper_ * cutoff_upper_, loop, include_transpose,
                            get_accessor<int32_t, 1>(i_curr_pair),
                            get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                            get_accessor<scalar_t, 1>(distances), use_periodic,
                            get_accessor<scalar_t, 2>(box_vectors));
                        if (loop) {
                            const uint64_t num_threads = 128;
                            const uint64_t num_blocks =
                                std::max((num_atoms + num_threads - 1ul) / num_threads, 1ul);
                            add_self_kernel<<<num_blocks, num_threads, 0, stream>>>(
                                num_atoms, get_accessor<scalar_t, 2>(positions),
                                get_accessor<int32_t, 1>(i_curr_pair),
                                get_accessor<int32_t, 2>(neighbors),
                                get_accessor<scalar_t, 2>(deltas),
                                get_accessor<scalar_t, 1>(distances));
                        }
                    });
            } else if (strat == strategy::shared) {
                AT_DISPATCH_FLOATING_TYPES(
                    positions.scalar_type(), "get_neighbor_pairs_shared_forward", [&]() {
                        const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
                        const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
                        TORCH_CHECK(cutoff_upper_ > 0, "Expected \"cutoff\" to be positive");
                        constexpr int BLOCKSIZE = 64;
                        const int num_blocks = std::max((num_atoms + BLOCKSIZE - 1) / BLOCKSIZE, 1);
                        const int num_threads = BLOCKSIZE;
                        const int num_tiles = num_blocks;
                        forward_kernel_shared<BLOCKSIZE><<<num_blocks, num_threads, 0, stream>>>(
                            num_atoms, get_accessor<scalar_t, 2>(positions),
                            get_accessor<int64_t, 1>(batch), cutoff_lower_ * cutoff_lower_,
                            cutoff_upper_ * cutoff_upper_, loop, include_transpose,
                            get_accessor<int32_t, 1>(i_curr_pair),
                            get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                            get_accessor<scalar_t, 1>(distances), num_tiles, use_periodic,
                            get_accessor<scalar_t, 2>(box_vectors));
                    });
            }
        }
        ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;
        return {neighbors, deltas, distances, i_curr_pair};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {
        return common_backward(ctx, grad_inputs);
    }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs_brute",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
               const tensor_list results = Autograd::apply(
                   positions, batch, cutoff_lower, cutoff_upper, box_vectors, use_periodic,
                   max_num_pairs, loop, include_transpose, strategy::brute);
               return std::make_tuple(results[0], results[1], results[2], results[3]);
           });
    m.impl("get_neighbor_pairs_shared",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
               const tensor_list results = Autograd::apply(
                   positions, batch, cutoff_lower, cutoff_upper, box_vectors, use_periodic,
                   max_num_pairs, loop, include_transpose, strategy::shared);
               return std::make_tuple(results[0], results[1], results[2], results[3]);
           });
}
