/* Raul P. Pelaez 2023. Common utilities for the CUDA neighbor operation.
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
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x){};
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
template <typename scalar_t> struct Box {
    scalar_t size[3][3];
    Box(const Tensor& box_vectors) {
        if (box_vectors.size(0) == 3 && box_vectors.size(1) == 3) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    size[i][j] = box_vectors[i][j].item<scalar_t>();
                }
            }
        }
    }
};
/*
 * @brief Takes a point to the unit cell using Minimum Image
 * Convention
 * @param p The point position
 * @param box_vectors The box vectors (3x3 matrix)
 * @return The point in the unit cell
 */
template <typename scalar_t>
__device__ auto apply_pbc(scalar3<scalar_t> delta, const Box<scalar_t>& box) {
    scalar_t scale3 = round(delta.z / box.size[2][2]);
    delta.x -= scale3 * box.size[2][0];
    delta.y -= scale3 * box.size[2][1];
    delta.z -= scale3 * box.size[2][2];
    scalar_t scale2 = round(delta.y / box.size[1][1]);
    delta.x -= scale2 * box.size[1][0];
    delta.y -= scale2 * box.size[1][1];
    scalar_t scale1 = round(delta.x / box.size[0][0]);
    delta.x -= scale1 * box.size[0][0];
    return delta;
}

template <typename scalar_t>
__device__ auto compute_distance(scalar3<scalar_t> pos_i, scalar3<scalar_t> pos_j,
                                 bool use_periodic, const Box<scalar_t>& box) {
    scalar3<scalar_t> delta = {pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z};
    if (use_periodic) {
        delta = apply_pbc(delta, box);
    }
    return delta;
}

} // namespace triclinic

/*
 * Backward pass for the CUDA neighbor list operation.
 * Computes the gradient of the positions with respect to the distances and deltas.
 */
tensor_list common_backward(AutogradContext* ctx, const tensor_list &grad_inputs);
#endif
