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
        auto r0 = edge_weight.nonzero().squeeze(-1);
        auto grad_positions = torch::zeros({num_atoms, 3}, edge_vec.options());
        // We need to avoid dividing by 0. Otherwise Autograd fills the gradient with NaNs in the
        // case of a double backwards. This is why we index_select like this.
        auto grad_distances_ =
            (edge_vec.index_select(0, r0) / edge_weight.index_select(0, r0).unsqueeze(-1)) *
            grad_edge_weight.index_select(0, r0).unsqueeze(-1);
        auto edge_index_no_r0 = edge_index.index_select(1, r0);
        auto result = grad_edge_vec.index_select(0, r0) + grad_distances_;
        grad_positions.index_add_(0, edge_index_no_r0[0], result);
        grad_positions.index_add_(0, edge_index_no_r0[1], -result);
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
