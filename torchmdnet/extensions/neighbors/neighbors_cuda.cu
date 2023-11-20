/* Raul P. Pelaez 2023
   Connection between the neighbor CUDA implementations and the torch extension.
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

    using Slice = torch::indexing::Slice;

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto edge_index = saved[0];
        auto edge_vec = saved[1];
        auto edge_weight = saved[2];
        auto num_atoms = ctx->saved_data["num_atoms"].toInt();
        auto grad_edge_vec = grad_outputs[1];
        auto grad_edge_weight = grad_outputs[2];
        auto zero_mask = edge_weight == 0;
        auto zero_mask3 = zero_mask.unsqueeze(-1).expand_as(grad_edge_vec);
        // We need to avoid dividing by 0. Otherwise Autograd fills the gradient with NaNs in the
        // case of a double backwards. This is why we index_select like this.
        auto grad_distances_ = edge_vec / edge_weight.masked_fill(zero_mask, 1).unsqueeze(-1) *
                               grad_edge_weight.masked_fill(zero_mask, 0).unsqueeze(-1);
        auto result = grad_edge_vec.masked_fill(zero_mask3, 0) + grad_distances_;
        // Given that there is no masked_index_add function, in order to make the operation
        // CUDA-graph compatible I need to transform masked indices into a dummy value (num_atoms)
        // and then exclude that value from the output.
	// TODO: replace this once masked_index_add  or masked_scatter_add are available
        auto grad_positions_ = torch::zeros({num_atoms + 1, 3}, edge_vec.options());
        auto edge_index_ =
            edge_index.masked_fill(zero_mask.unsqueeze(0).expand_as(edge_index), num_atoms);
        grad_positions_.index_add_(0, edge_index_[0], result);
        grad_positions_.index_add_(0, edge_index_[1], -result);
        auto grad_positions = grad_positions_.index({Slice(0, num_atoms), Slice()});
        Tensor ignore;
        return {ignore, grad_positions, ignore, ignore, ignore, ignore,
                ignore, ignore,         ignore, ignore, ignore};
    }
};

TORCH_LIBRARY_IMPL(torchmdnet_extensions, AutogradCUDA, m) {
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
