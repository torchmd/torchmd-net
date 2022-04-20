#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;
using torch::kInt32;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;
using torch::Tensor;
using torch::TensorOptions;

template <typename scalar_t, int num_dims>
    using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims> 
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __global__ void kernel(
    const Accessor<int32_t, 2> neighbors,
    const Accessor<scalar_t, 2> source,
    Accessor<scalar_t, 2> destination
) {
    const int32_t i_neig = blockIdx.x;
    if (i_neig >= neighbors.size(0)) return;

    const int32_t i_src = neighbors[blockIdx.y][i_neig];
    if (i_src < 0) return;

    const int32_t i_dest = neighbors[!blockIdx.y][i_neig];
    if (i_dest < 0) return;

    const int32_t i_feat = threadIdx.x;
    atomicAdd(&destination[i_dest][i_feat], source[i_src][i_feat]);
}

static Tensor run_kernel(const Tensor& neighbors, const Tensor& source) {

    const int num_neighbors = neighbors.size(0);
    const int num_features = source.size(1);

    const dim3 blocks(num_neighbors, 2);
    const dim3 threads(num_features);
    const auto stream = getCurrentCUDAStream(neighbors.get_device());

    Tensor destination = source.clone();

    AT_DISPATCH_FLOATING_TYPES(source.scalar_type(), "pass_messages", [&]() {
        const CUDAStreamGuard guard(stream);
        kernel<<<blocks, threads, 0, stream>>>(
            get_accessor<int32_t, 2>(neighbors),
            get_accessor<scalar_t, 2>(source),
            get_accessor<scalar_t, 2>(destination));
    });

    return destination;
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& neighbors, const Tensor& messages) {

        TORCH_CHECK(neighbors.dim() == 2, "Expected \"neighbors\" to have two dimensions");
        TORCH_CHECK(neighbors.size(0) == 2, "Expected the 2nd dimension size of \"neighbors\" to be 2");
        TORCH_CHECK(neighbors.scalar_type() == kInt32, "Expected \"neighbors\" to have data type of int32");
        TORCH_CHECK(neighbors.is_contiguous(), "Expected \"neighbors\" to be contiguous");

        TORCH_CHECK(messages.dim() == 2, "Expected \"messages\" to have two dimensions");
        TORCH_CHECK(messages.size(1) % 32 == 0, "Expected the 2nd dimension size of \"messages\" to be a multiple of 32");
        TORCH_CHECK(messages.size(1) <= 1024, "Expected the 2nd dimension size of \"messages\" to be less than 1024");
        TORCH_CHECK(messages.is_contiguous(), "Expected \"messages\" to be contiguous");

        ctx->save_for_backward({neighbors});

        return {run_kernel(neighbors, messages)};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {

        const Tensor neighbors = ctx->get_saved_variables()[0];
        const Tensor grad_messages = grad_inputs[0];

        return {Tensor(), run_kernel(neighbors, grad_messages)};
    }
};

TORCH_LIBRARY_IMPL(messages, AutogradCUDA, m) {
    m.impl("pass_messages", [](const Tensor& neighbors, const Tensor& messages) {
        return Autograd::apply(neighbors, messages)[0];
    });
}