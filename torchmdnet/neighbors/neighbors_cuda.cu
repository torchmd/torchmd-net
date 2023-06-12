/* Raul P. Pelaez 2023
   Connection between the neighbor CUDA implementations and the torch extension.
   See neighbors.cpp for the definition of the torch extension functions.
 */
#include "neighbors_cuda_brute.cuh"
#include "neighbors_cuda_cell.cuh"
#include "neighbors_cuda_shared.cuh"
#include <torch/extension.h>
template <class... T> auto call_forward_kernel(const std::string& kernel_name, const T&... args) {
    if (kernel_name == "brute") {
        return forward_brute(args...);
    } else if (kernel_name == "cell") {
        return forward_cell(args...);
    } else if (kernel_name == "shared") {
        return forward_shared(args...);
    } else {
        throw std::runtime_error("Unknown kernel name");
    }
}

template <typename scalar_t>
__global__ void masked_index_add_cuda_forward(const Accessor<scalar_t, 2> values,
                                              const Accessor<int, 1> index,
                                              Accessor<scalar_t, 2> output) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < index.size(0)) {
        const auto k = index[i];
        if (k >= 0) {
            for (int j = 0; j < values.size(1); ++j) {
                atomicAdd(&output[k][j], values[i][j]);
            }
        }
    }
}

template <typename scalar_t>
__global__ void masked_index_add_cuda_backward(Accessor<scalar_t, 2> grad_input,
                                               const Accessor<int, 1> index,
                                               const Accessor<scalar_t, 2> values,
                                               const Accessor<scalar_t, 2> grad_output) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < index.size(0)) {
        const auto k = index[i];
        if (k >= 0) {
            for (int j = 0; j < grad_input.size(1); ++j) {
                // Gradients must be accmulated for the output and the values
                atomicAdd(&grad_input[i][j], grad_output[k][j]);
            }
        }
    }
}

/*
 *The MaskedIndexAdd operation works similarly as index_add from pytorch, but it only adds the
 *elements of the values tensor to the input if the index is positive.
 */
class MaskedIndexAddFunction : public torch::autograd::Function<MaskedIndexAddFunction> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor &input,const  Tensor &index,const  Tensor& values) {
        ctx->save_for_backward({index, values});
        ctx->saved_data["input_size"] = input.size(0);
        auto stream = at::cuda::getCurrentCUDAStream();
        const CUDAStreamGuard guard(stream);
        auto output = input.clone();
        const int threads = 128;
        const dim3 blocks((index.numel() + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "masked_index_add_forward_cuda", ([&] {
                masked_index_add_cuda_forward<scalar_t><<<blocks, threads, 0, stream>>>(
                    get_accessor<scalar_t, 2>(values), get_accessor<int, 1>(index),
                    get_accessor<scalar_t, 2>(output));
            }));
        return {output};
    }

    static tensor_list backward(AutogradContext* ctx, const tensor_list & grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto index = saved[0];
        auto values = saved[1];
        auto input_size = ctx->saved_data["input_size"].toInt();
        auto grad_output = grad_outputs[0];
        auto grad_input = torch::rand_like(grad_output);
	  //zeros_like(grad_output);
        auto stream = at::cuda::getCurrentCUDAStream();
        const CUDAStreamGuard guard(stream);
        const int threads = 128;
        const dim3 blocks((index.numel() + threads - 1) / threads);
	std::cout << "grad_input before: " << grad_input << std::endl;
        AT_DISPATCH_FLOATING_TYPES(
            grad_output.scalar_type(), "masked_index_add_backward_cuda", ([&] {
                masked_index_add_cuda_backward<scalar_t><<<blocks, threads, 0, stream>>>(
                    get_accessor<scalar_t, 2>(grad_input), get_accessor<int, 1>(index),
                    get_accessor<scalar_t, 2>(values), get_accessor<scalar_t, 2>(grad_output));
            }));
	std::cout << "grad_input after: " << grad_input << std::endl;
	std::cout << "grad_output: " << grad_output << std::endl;
        return {grad_input, Tensor(), Tensor()};
    }
};

// This is the autograd function that is called when the user calls get_neighbor_pairs.
// It dispatches the required strategy for the forward function and implements the backward
// function. The backward function is written in full pytorch so that it can be differentiated a
// second time automatically via Autograd.
class NeighborAutograd : public torch::autograd::Function<NeighborAutograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const std::string& strategy,
                               const Tensor& positions, const Tensor& batch,
                               const Tensor& box_vectors, bool use_periodic,
                               const Scalar& cutoff_lower, const Scalar& cutoff_upper,
                               const Scalar& max_num_pairs, bool loop, bool include_transpose) {
        Tensor neighbors, deltas, distances, i_curr_pair;
        std::tie(neighbors, deltas, distances, i_curr_pair) =
            call_forward_kernel(strategy, positions, batch, box_vectors, use_periodic, cutoff_lower,
                                cutoff_upper, max_num_pairs, loop, include_transpose);
        ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = positions.size(0);
        return {neighbors, deltas, distances, i_curr_pair};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto edge_index = saved[0];
        auto edge_vec = saved[1];
        auto edge_weight = saved[2];
        auto num_atoms = ctx->saved_data["num_atoms"].toInt();
        auto grad_edge_vec = grad_outputs[1];
        auto grad_edge_weight = grad_outputs[2];
        auto grad_positions = torch::zeros({num_atoms, 3}, edge_vec.options());
        auto r0 = edge_weight.eq(0);
        auto edge_weight_ = edge_weight.masked_fill(r0, 1);
        auto grad_distances_ = (edge_vec / edge_weight_.unsqueeze(-1).expand_as(edge_vec)) *
                               grad_edge_weight.unsqueeze(-1).expand_as(edge_vec);
        auto result = grad_edge_vec + grad_distances_;
        grad_positions = MaskedIndexAddFunction::apply(grad_positions, edge_index[0], result)[0];
        grad_positions = MaskedIndexAddFunction::apply(grad_positions, edge_index[1], -result)[0];
        Tensor ignore;
        return {ignore, grad_positions, ignore, ignore, ignore, ignore,
                ignore, ignore,         ignore, ignore, ignore};
    }
};

TORCH_LIBRARY_IMPL(torchmdnet_neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs",
           [](const std::string& strategy, const Tensor& positions, const Tensor& batch,
              const Tensor& box_vectors, bool use_periodic, const Scalar& cutoff_lower,
              const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
              bool include_transpose) {
               auto final_strategy = strategy;
               if (positions.size(0) >= 32768 && strategy == "brute") {
                   final_strategy = "shared";
               }
               auto result = NeighborAutograd::apply(final_strategy, positions, batch, box_vectors,
                                                     use_periodic, cutoff_lower, cutoff_upper,
                                                     max_num_pairs, loop, include_transpose);
               return std::make_tuple(result[0], result[1], result[2], result[3]);
           });
}
