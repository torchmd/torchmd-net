/* Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 *(See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 *
 * Raul P. Pelaez 2023. Brute force neighbor list construction in CUDA.
 *
 * A brute force approach that assigns a thread per each possible pair of particles in the system.
 * Based on an implementation by Raimondas Galvelis.
 * Works fantastically for small (less than 10K atoms) systems, but cannot handle more than 32K
 * atoms.
 */
#ifndef NEIGHBORS_BRUTE_CUH
#define NEIGHBORS_BRUTE_CUH
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
                                     scalar_t cutoff_upper2, PairListAccessor<scalar_t> list,
                                     triclinic::BoxAccessor<scalar_t> box) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_all_pairs)
        return;
    const uint32_t row = get_row(index);
    const uint32_t column = (index - row * (row - 1) / 2);
    const auto batch_row = batch[row];
    if (batch_row == batch[column]) {
        const auto pos_i = fetchPosition(positions, row);
        const auto pos_j = fetchPosition(positions, column);
        const auto box_row = box[batch_row];
        const auto delta = triclinic::compute_distance(pos_i, pos_j, list.use_periodic, box_row);
        const scalar_t distance2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (distance2 < cutoff_upper2 && distance2 >= cutoff_lower2) {
            const scalar_t r2 = sqrt_(distance2);
            addAtomPairToList(list, row, column, delta, r2, list.include_transpose);
        }
    }
}

template <typename scalar_t>
__global__ void add_self_kernel(const int num_atoms, Accessor<scalar_t, 2> positions,
                                PairListAccessor<scalar_t> list) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    __shared__ int i_pair;
    if (threadIdx.x == 0) { // Each block adds blockDim.x pairs to the list.
        // Handle the last block, so that only num_atoms are added in total
        i_pair =
            atomicAdd(&list.i_curr_pair[0], min(blockDim.x, num_atoms - blockIdx.x * blockDim.x));
    }
    __syncthreads();
    scalar3<scalar_t> delta{};
    scalar_t distance = 0;
    writeAtomPair(list, i_atom, i_atom, delta, distance, i_pair + threadIdx.x);
}

static std::tuple<Tensor, Tensor, Tensor, Tensor>
forward_brute(const Tensor& positions, const Tensor& batch, const Tensor& in_box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
    checkInput(positions, batch);
    const auto max_num_pairs_ = max_num_pairs.toLong();
    TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
    auto box_vectors = in_box_vectors.to(positions.device()).clone();
    if (box_vectors.dim() == 2) {
        // If the box is a 3x3 tensor it is assumed every sample has the same box
        if (use_periodic) {
            TORCH_CHECK(box_vectors.size(0) == 3 && box_vectors.size(1) == 3,
                        "Expected \"box_vectors\" to have shape (3, 3)");
        }
        // Make the box (None,3,3), expand artificially to positions.size(0)
        box_vectors = box_vectors.unsqueeze(0);
        if (use_periodic) {
            // I use positions.size(0) because the batch dimension is not available here
            box_vectors = box_vectors.expand({positions.size(0), 3, 3});
        }
    }
    if (use_periodic) {
        TORCH_CHECK(box_vectors.dim() == 3, "Expected \"box_vectors\" to have three dimensions");
        TORCH_CHECK(box_vectors.size(1) == 3 && box_vectors.size(2) == 3,
                    "Expected \"box_vectors\" to have shape (n_batch, 3, 3)");
    }
    const int num_atoms = positions.size(0);
    TORCH_CHECK(num_atoms < 32768, "The brute strategy fails with \"num_atoms\" larger than 32768");
    const int num_pairs = max_num_pairs_;
    const TensorOptions options = positions.options();
    const auto stream = getCurrentCUDAStream(positions.get_device());
    PairList list(num_pairs, positions.options(), loop, include_transpose, use_periodic);
    const CUDAStreamGuard guard(stream);
    const uint64_t num_all_pairs = num_atoms * (num_atoms - 1UL) / 2UL;
    const uint64_t num_threads = 128;
    const uint64_t num_blocks = std::max((num_all_pairs + num_threads - 1UL) / num_threads, 1UL);
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "get_neighbor_pairs_forward", [&]() {
        PairListAccessor<scalar_t> list_accessor(list);
        auto box = triclinic::get_box_accessor<scalar_t>(box_vectors, use_periodic);
        const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
        const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
        TORCH_CHECK(cutoff_upper_ > 0, "Expected \"cutoff\" to be positive");
        forward_kernel_brute<<<num_blocks, num_threads, 0, stream>>>(
            num_all_pairs, get_accessor<scalar_t, 2>(positions), get_accessor<int64_t, 1>(batch),
            cutoff_lower_ * cutoff_lower_, cutoff_upper_ * cutoff_upper_, list_accessor, box);
        if (loop) {
            const uint32_t num_threads_self = 256;
            const uint32_t num_blocks_self =
                std::max((num_atoms + num_threads_self - 1U) / num_threads_self, 1U);
            add_self_kernel<<<num_blocks_self, num_threads_self, 0, stream>>>(
                num_atoms, get_accessor<scalar_t, 2>(positions), list_accessor);
        }
    });
    return {list.neighbors, list.deltas, list.distances, list.i_curr_pair};
}

#endif
