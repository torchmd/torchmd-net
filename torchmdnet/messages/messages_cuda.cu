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

template <typename scalar_t> __global__ void kernel_forward(
    const Accessor<int32_t, 2> neighbors,
    const Accessor<scalar_t, 2> messages,
    Accessor<scalar_t, 2> new_states
) {
    const int32_t i_neig = blockIdx.x;
    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_neig];
    if (i_atom < 0) return;

    const int32_t i_feat = threadIdx.x;
    atomicAdd(&new_states[i_atom][i_feat], messages[i_neig][i_feat]);
}

template <typename scalar_t> __global__ void kernel_backward(
    const Accessor<int32_t, 2> neighbors,
    const Accessor<scalar_t, 2> grad_new_state,
    Accessor<scalar_t, 2> grad_messages
) {
    const int32_t i_neig = blockIdx.x;
    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_neig];
    if (i_atom < 0) return;

    const int32_t i_feat = threadIdx.x;
    atomicAdd(&grad_messages[i_neig][i_feat], grad_new_state[i_atom][i_feat]);
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx,
                               const Tensor& neighbors,
                               const Tensor& messages,
                               const Tensor& states) {

        TORCH_CHECK(neighbors.dim() == 2, "Expected \"neighbors\" to have two dimensions");
        TORCH_CHECK(neighbors.size(0) == 2, "Expected the 2nd dimension size of \"neighbors\" to be 2");
        TORCH_CHECK(neighbors.scalar_type() == kInt32, "Expected \"neighbors\" to have data type of int32");
        TORCH_CHECK(neighbors.is_contiguous(), "Expected \"neighbors\" to be contiguous");

        TORCH_CHECK(messages.dim() == 2, "Expected \"messages\" to have two dimensions");
        TORCH_CHECK(messages.size(1) % 32 == 0, "Expected the 2nd dimension size of \"messages\" to be a multiple of 32");
        TORCH_CHECK(messages.size(1) <= 1024, "Expected the 2nd dimension size of \"messages\" to be less than 1024");
        TORCH_CHECK(messages.is_contiguous(), "Expected \"messages\" to be contiguous");

        TORCH_CHECK(states.dim() == 2, "Expected \"states\" to have two dimensions");
        TORCH_CHECK(states.size(1) == messages.size(1), "Expected the 2nd dimension size of \"messages\" and \"states\" to be the same");
        TORCH_CHECK(states.scalar_type() == messages.scalar_type(), "Expected the data type of \"messages\" and \"states\" to be the same");
        TORCH_CHECK(states.is_contiguous(), "Expected \"messages\" to be contiguous");

        const int num_neighbors = neighbors.size(1);
        const int num_features = messages.size(1);

        const dim3 blocks(num_neighbors, 2);
        const dim3 threads(num_features);
        const auto stream = getCurrentCUDAStream(neighbors.get_device());

        Tensor new_states = states.clone();

        AT_DISPATCH_FLOATING_TYPES(messages.scalar_type(), "pass_messages_forward", [&]() {
            const CUDAStreamGuard guard(stream);
            kernel_forward<<<blocks, threads, 0, stream>>>(
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(messages),
                get_accessor<scalar_t, 2>(new_states));
        });

        ctx->save_for_backward({neighbors});

        return {new_states};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {

        const Tensor neighbors = ctx->get_saved_variables()[0];
        const Tensor grad_new_state = grad_inputs[0];

        const int num_neighbors = neighbors.size(1);
        const int num_features = grad_new_state.size(1);

        const dim3 blocks(num_neighbors, 2);
        const dim3 threads(num_features);
        const auto stream = getCurrentCUDAStream(neighbors.get_device());

        Tensor grad_messages = torch::zeros({num_neighbors, num_features}, grad_new_state.options());

        AT_DISPATCH_FLOATING_TYPES(grad_new_state.scalar_type(), "pass_messages_backward", [&]() {
            const CUDAStreamGuard guard(stream);
            kernel_backward<<<blocks, threads, 0, stream>>>(
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(grad_new_state),
                get_accessor<scalar_t, 2>(grad_messages));
        });

        return {Tensor(), // grad_neighbors
                grad_messages,
                grad_new_state.clone()}; // grad_state
    }
};

TORCH_LIBRARY_IMPL(messages, AutogradCUDA, m) {
    m.impl("pass_messages", [](const Tensor& neighbors,
                               const Tensor& messages,
                               const Tensor& states) {
        return Autograd::apply(neighbors, messages, states)[0];
    });
}