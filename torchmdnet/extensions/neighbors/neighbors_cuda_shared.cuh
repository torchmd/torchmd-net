/* Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 *(See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 * Raul P. Pelaez 2023. Shared memory neighbor list construction for CUDA.
 * This brute force approach checks all pairs of atoms by collaborativelly loading and processing
 * tiles of atoms into shared memory.
 * This approach is tipically slower than the brute force approach, but can handle an arbitrarily
 * large number of atoms.
 */
#ifndef NEIGHBORS_SHARED_CUH
#define NEIGHBORS_SHARED_CUH
#include "common.cuh"
#include <algorithm>
#include <torch/extension.h>

template <int BLOCKSIZE, typename scalar_t>
__global__ void forward_kernel_shared(uint32_t num_atoms, const Accessor<scalar_t, 2> positions,
                                      const Accessor<int64_t, 1> batch, scalar_t cutoff_lower2,
                                      scalar_t cutoff_upper2, PairListAccessor<scalar_t> list,
                                      int32_t num_tiles, triclinic::BoxAccessor<scalar_t> box) {
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
        pos_i = fetchPosition(positions, id);
        batch_i = batch[id];
    }
    // Distribute the N particles in a group of tiles. Storing in each tile blockDim.x values in
    // shared memory. This way all threads are accesing the same memory addresses at the same time
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load this tiles particles values to shared memory
        const int i_load = tile * blockDim.x + threadIdx.x;
        if (i_load < num_atoms) { // Even if im not active, my thread may load a value each tile to
                                  // shared memory.
            sh_pos[threadIdx.x] = fetchPosition(positions, i_load);
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
            const bool testPair = cur_j < num_atoms and (cur_j < id or (list.loop and cur_j == id));
            if (testPair) {
                const auto batch_j = sh_batch[counter];
                if (batch_i == batch_j) {
                    const auto pos_j = sh_pos[counter];
                    const auto box_i = box[batch_i];
                    const auto delta =
                        triclinic::compute_distance(pos_i, pos_j, list.use_periodic, box_i);
                    const scalar_t distance2 =
                        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                    if (distance2 < cutoff_upper2 && distance2 >= cutoff_lower2) {
                        const bool requires_transpose = list.include_transpose && !(cur_j == id);
                        const auto distance = sqrt_(distance2);
                        addAtomPairToList(list, id, cur_j, delta, distance, requires_transpose);
                    }
                }
            }
        }
        __syncthreads();
    }
}

static std::tuple<Tensor, Tensor, Tensor, Tensor>
forward_shared(const Tensor& positions, const Tensor& batch, const Tensor& in_box_vectors,
               bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
               const Scalar& max_num_pairs, bool loop, bool include_transpose) {
    checkInput(positions, batch);
    const auto max_num_pairs_ = max_num_pairs.toLong();
    auto box_vectors = in_box_vectors.to(positions.device());
    if (box_vectors.dim() == 2) {
        // If the box is a 3x3 tensor it is assumed every sample has the same box
        if (use_periodic) {
            TORCH_CHECK(box_vectors.size(0) == 3 && box_vectors.size(1) == 3,
                        "Expected \"box_vectors\" to have shape (n_batch, 3, 3)");
        }
        // Make the box (None,3,3), expand artificially to positions.size(0)
        box_vectors = box_vectors.unsqueeze(0);
        if (use_periodic) {
            // I use positions.size(0) because the batch dimension is not available here
            box_vectors = box_vectors.expand({positions.size(0), 3, 3});
        }
    }
    TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
    if (use_periodic) {
        TORCH_CHECK(box_vectors.dim() == 3, "Expected \"box_vectors\" to have three dimensions");
        TORCH_CHECK(box_vectors.size(1) == 3 && box_vectors.size(2) == 3,
                    "Expected \"box_vectors\" to have shape (n_batch, 3, 3)");
    }
    const int num_atoms = positions.size(0);
    const int num_pairs = max_num_pairs_;
    const TensorOptions options = positions.options();
    const auto stream = getCurrentCUDAStream(positions.get_device());
    PairList list(num_pairs, positions.options(), loop, include_transpose, use_periodic);
    const CUDAStreamGuard guard(stream);
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_pairs_shared_forward", [&]() {
        const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
        const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
        auto box = triclinic::get_box_accessor<scalar_t>(box_vectors, use_periodic);
        TORCH_CHECK(cutoff_upper_ > 0, "Expected \"cutoff\" to be positive");
        constexpr int BLOCKSIZE = 64;
        const int num_blocks = std::max((num_atoms + BLOCKSIZE - 1) / BLOCKSIZE, 1);
        const int num_threads = BLOCKSIZE;
        const int num_tiles = num_blocks;
        PairListAccessor<scalar_t> list_accessor(list);
        forward_kernel_shared<BLOCKSIZE><<<num_blocks, num_threads, 0, stream>>>(
            num_atoms, get_accessor<scalar_t, 2>(positions), get_accessor<int64_t, 1>(batch),
            cutoff_lower_ * cutoff_lower_, cutoff_upper_ * cutoff_upper_, list_accessor, num_tiles,
            box);
    });
    return {list.neighbors, list.deltas, list.distances, list.i_curr_pair};
}

#endif
