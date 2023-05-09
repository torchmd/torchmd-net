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

template <typename scalar_t> struct vec4 {
    using type = void;
};
template <> struct vec4<float> {
    using type = float4;
};
template <> struct vec4<double> {
    using type = double4;
};

template <typename scalar_t> using scalar4 = typename vec4<scalar_t>::type;

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
/*
 * @brief Takes a point to the unit cell using Minimum Image
 * Convention
 * @param p The point position
 * @param box_vectors The box vectors (3x3 matrix)
 * @return The point in the unit cell
 */
template <typename scalar_t>
__device__ auto apply_pbc(scalar3<scalar_t> delta, const Accessor<scalar_t, 2> box_vectors) {
    scalar_t scale3 = round(delta.z / box_vectors[2][2]);
    delta.x -= scale3 * box_vectors[2][0];
    delta.y -= scale3 * box_vectors[2][1];
    delta.z -= scale3 * box_vectors[2][2];
    scalar_t scale2 = round(delta.y / box_vectors[1][1]);
    delta.x -= scale2 * box_vectors[1][0];
    delta.y -= scale2 * box_vectors[1][1];
    scalar_t scale1 = round(delta.x / box_vectors[0][0]);
    delta.x -= scale1 * box_vectors[0][0];
    return delta;
}

template <typename scalar_t>
__device__ auto compute_distance(scalar3<scalar_t> pos_i, scalar3<scalar_t> pos_j,
                                 bool use_periodic, const Accessor<scalar_t, 2> box_vectors) {
    scalar3<scalar_t> delta = {pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z};
    if (use_periodic) {
        delta = apply_pbc(delta, box_vectors);
    }
    return delta;
}

} // namespace triclinic

tensor_list common_backward(AutogradContext* ctx, tensor_list grad_inputs);
#endif
