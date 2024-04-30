/* Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 *(See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 * Raul P. Pelaez 2023. Common utilities for the CUDA neighbor operation.
 */
#ifndef NEIGHBORS_COMMON_CUH
#define NEIGHBORS_COMMON_CUH
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using torch::empty;
using torch::full;
using torch::kInt32;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;
using torch::zeros;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <typename scalar_t, int num_dims>
using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dims, torch::RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
using KernelAccessor = at::TensorAccessor<scalar_t, num_dims, at::RestrictPtrTraits, signed int>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x){return ::sqrt(x);};
template <> __device__ __forceinline__ float sqrt_(float x) {
    return ::sqrtf(x);
};
template <> __device__ __forceinline__ double sqrt_(double x) {
    return ::sqrt(x);
};

template <typename scalar_t> struct vec3 {
    using type = void;
};

template <> struct vec3<float> {
    using type = float3;
};

template <> struct vec3<double> {
    using type = double3;
};

template <typename scalar_t> using scalar3 = typename vec3<scalar_t>::type;

/*
 * @brief Get the position of the i'th particle
 * @param positions The positions tensor
 * @param i The index of the particle
 * @return The position of the i'th particle
 */
template <class scalar_t>
__device__ scalar3<scalar_t> fetchPosition(const Accessor<scalar_t, 2> positions, const int i) {
    return {positions[i][0], positions[i][1], positions[i][2]};
}

struct PairList {
    Tensor i_curr_pair;
    Tensor neighbors;
    Tensor deltas;
    Tensor distances;
    const bool loop, include_transpose, use_periodic;
    PairList(int max_num_pairs, TensorOptions options, bool loop, bool include_transpose,
             bool use_periodic)
        : i_curr_pair(zeros({1}, options.dtype(torch::kInt))),
          neighbors(full({2, max_num_pairs}, -1, options.dtype(torch::kInt))),
          deltas(full({max_num_pairs, 3}, 0, options)),
          distances(full({max_num_pairs}, 0, options)), loop(loop),
          include_transpose(include_transpose), use_periodic(use_periodic) {
    }
};

template <class scalar_t> struct PairListAccessor {
    Accessor<int32_t, 1> i_curr_pair;
    Accessor<int32_t, 2> neighbors;
    Accessor<scalar_t, 2> deltas;
    Accessor<scalar_t, 1> distances;
    bool loop, include_transpose, use_periodic;
    explicit PairListAccessor(const PairList& pl)
        : i_curr_pair(get_accessor<int32_t, 1>(pl.i_curr_pair)),
          neighbors(get_accessor<int32_t, 2>(pl.neighbors)),
          deltas(get_accessor<scalar_t, 2>(pl.deltas)),
          distances(get_accessor<scalar_t, 1>(pl.distances)), loop(pl.loop),
          include_transpose(pl.include_transpose), use_periodic(pl.use_periodic) {
    }
};

template <typename scalar_t>
__device__ void writeAtomPair(PairListAccessor<scalar_t>& list, int i, int j,
                              scalar3<scalar_t> delta, scalar_t distance, int i_pair) {
    if (i_pair < list.neighbors.size(1)) {
        list.neighbors[0][i_pair] = i;
        list.neighbors[1][i_pair] = j;
        list.deltas[i_pair][0] = delta.x;
        list.deltas[i_pair][1] = delta.y;
        list.deltas[i_pair][2] = delta.z;
        list.distances[i_pair] = distance;
    }
}

template <typename scalar_t>
__device__ void addAtomPairToList(PairListAccessor<scalar_t>& list, int i, int j,
                                  scalar3<scalar_t> delta, scalar_t distance, bool add_transpose) {
    const int32_t i_pair = atomicAdd(&list.i_curr_pair[0], add_transpose ? 2 : 1);
    // Neighbors after the max number of pairs are ignored, although the pair is counted
    writeAtomPair(list, i, j, delta, distance, i_pair);
    if (add_transpose) {
        writeAtomPair(list, j, i, {-delta.x, -delta.y, -delta.z}, distance, i_pair + 1);
    }
}

static void checkInput(const Tensor& positions, const Tensor& batch) {
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
    TORCH_CHECK(batch.dtype() == torch::kInt64, "Expected \"batch\" to be of type torch::kLong");
}

namespace rect {

/*
 * @brief Takes a point to the unit cell in the range [-0.5, 0.5]*box_size using Minimum Image
 * Convention
 * @param p The point position
 * @param box_size The box size
 * @return The point in the unit cell
 */
template <typename scalar_t>
__device__ auto apply_pbc(scalar3<scalar_t> p, scalar3<scalar_t> box_size) {
    p.x = p.x - floorf(p.x / box_size.x + scalar_t(0.5)) * box_size.x;
    p.y = p.y - floorf(p.y / box_size.y + scalar_t(0.5)) * box_size.y;
    p.z = p.z - floorf(p.z / box_size.z + scalar_t(0.5)) * box_size.z;
    return p;
}

template <typename scalar_t>
__device__ auto compute_distance(scalar3<scalar_t> pos_i, scalar3<scalar_t> pos_j,
                                 bool use_periodic, scalar3<scalar_t> box_size) {
    scalar3<scalar_t> delta = {pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z};
    if (use_periodic) {
        delta = apply_pbc<scalar_t>(delta, box_size);
    }
    return delta;
}

} // namespace rect

namespace triclinic {
template <class scalar_t> using BoxAccessor = Accessor<scalar_t, 3>;
template <typename scalar_t>
BoxAccessor<scalar_t> get_box_accessor(const Tensor& box_vectors, bool use_periodic) {
    return get_accessor<scalar_t, 3>(box_vectors);
}

/*
 * @brief Takes a point to the unit cell using Minimum Image
 * Convention
 * @param p The point position
 * @param box_vectors The box vectors (3x3 matrix)
 * @return The point in the unit cell
 */
template <typename scalar_t>
__device__ auto apply_pbc(scalar3<scalar_t> delta, const KernelAccessor<scalar_t, 2>& box) {
    scalar_t scale3 = round(delta.z / box[2][2]);
    delta.x -= scale3 * box[2][0];
    delta.y -= scale3 * box[2][1];
    delta.z -= scale3 * box[2][2];
    scalar_t scale2 = round(delta.y / box[1][1]);
    delta.x -= scale2 * box[1][0];
    delta.y -= scale2 * box[1][1];
    scalar_t scale1 = round(delta.x / box[0][0]);
    delta.x -= scale1 * box[0][0];
    return delta;
}

  template <typename scalar_t>
__device__ auto compute_distance(scalar3<scalar_t> pos_i, scalar3<scalar_t> pos_j,
                                 bool use_periodic, const KernelAccessor<scalar_t, 2>& box) {
    scalar3<scalar_t> delta = {pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z};
    if (use_periodic) {
      delta = apply_pbc(delta, box);
    }
    return delta;
}

} // namespace triclinic

#endif
