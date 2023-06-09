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
__global__ void
forward_kernel(const Accessor<int32_t, 2> neighbors, const Accessor<scalar_t, 2> deltas,
               const Accessor<scalar_t, 2> grad_deltas, const Accessor<scalar_t, 1> distances,
               const Accessor<scalar_t, 1> grad_distances, Accessor<scalar_t, 2> grad_positions) {
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs) {
        return;
    }
    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    const int32_t i_comp = blockIdx.z;
    if (i_atom < 0) {
        return;
    }
    const scalar_t grad_deltas_ = grad_deltas[i_pair][i_comp];
    const scalar_t dist = distances[i_pair];
    const scalar_t grad_distances_ =
        dist == 0 ? scalar_t(1.0) : (deltas[i_pair][i_comp] / dist * grad_distances[i_pair]);

    // Handle self interaction
    const scalar_t grad =
        (i_dir ? -1 : 1) *
        (i_atom == neighbors[1 - i_dir][i_pair] ? scalar_t(0.0) : (grad_deltas_ + grad_distances_));
    atomicAdd(&grad_positions[i_atom][i_comp], grad);
}
template <typename scalar_t>
__global__ void
backward_kernel(const Accessor<int32_t, 2> neighbors, const Accessor<scalar_t, 2> deltas,
                const Accessor<scalar_t, 2> grad_deltas, const Accessor<scalar_t, 1> distances,
                const Accessor<scalar_t, 1> grad_distances, Accessor<scalar_t, 2> grad_positions,
                Accessor<scalar_t, 2> grad_edge_vec, Accessor<scalar_t, 1> grad_edge_weight) {
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs) {
        return;
    }
    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    const int32_t i_comp = blockIdx.z;
    if (i_atom < 0) {
        return;
    }

    scalar_t grad = grad_positions[i_atom][i_comp];
    if (distances[i_pair] != 0) {
        grad_edge_vec[i_pair][i_comp] = grad_distances[i_pair] * grad / distances[i_pair];
    }
    grad_edge_weight[i_pair] = grad_deltas[i_pair][i_comp] * grad;
}

class EdgeOperation : public torch::autograd::Function<EdgeOperation> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& edge_index,
                               const Tensor& edge_vec, const Tensor& edge_weight,
                               const Tensor& grad_edge_vec, const Tensor& grad_edge_weight) {
        ctx->save_for_backward(
            {edge_index, edge_vec, edge_weight, grad_edge_vec, grad_edge_weight});
        const Tensor& grad_deltas = grad_edge_vec;
        const Tensor& grad_distances = grad_edge_weight;
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks_x = std::max((num_pairs + num_threads - 1) / num_threads, 1);
        const dim3 blocks(num_blocks_x, 2, 3);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        const tensor_list data = ctx->get_saved_variables();
        const Tensor& neighbors = data[0];
        const Tensor& deltas = data[1];
        const Tensor& distances = data[2];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(
            grad_distances.scalar_type(), "getNeighborPairs::backward", [&]() {
                const CUDAStreamGuard guard(stream);
                forward_kernel<<<blocks, num_threads, 0, stream>>>(
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 2>(grad_deltas), get_accessor<scalar_t, 1>(distances),
                    get_accessor<scalar_t, 1>(grad_distances),
                    get_accessor<scalar_t, 2>(grad_positions));
            });

        return {grad_positions, Tensor(), Tensor(), Tensor(), Tensor(),
                Tensor(),       Tensor(), Tensor(), Tensor(), Tensor()};
    }
    static tensor_list backward(AutogradContext* ctx, const tensor_list& grad_outputs) {
        const tensor_list data = ctx->get_saved_variables();
        const Tensor& neighbors = data[0];
        const Tensor& deltas = data[1];
        const Tensor& distances = data[2];
        const Tensor& grad_positions = grad_outputs[0];
        const Tensor& grad_deltas = data[3];
        const Tensor& grad_distances = data[4];

        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks_x = std::max((num_pairs + num_threads - 1) / num_threads, 1);
        const dim3 blocks(num_blocks_x, 2, 3);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        Tensor grad_grad_edge_vec = zeros_like(deltas);
        Tensor grad_grad_edge_weight = zeros_like(distances);

        AT_DISPATCH_FLOATING_TYPES(
            grad_positions.scalar_type(), "getNeighborPairs::backward", [&]() {
                const CUDAStreamGuard guard(stream);
                backward_kernel<<<blocks, num_threads, 0, stream>>>(
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 2>(grad_deltas), get_accessor<scalar_t, 1>(distances),
		    get_accessor<scalar_t, 1>(grad_distances),
                    get_accessor<scalar_t, 2>(grad_positions),
                    get_accessor<scalar_t, 2>(grad_grad_edge_vec),
                    get_accessor<scalar_t, 1>(grad_grad_edge_weight));
            });

        return {Tensor(), Tensor(), Tensor(), grad_grad_edge_vec, grad_grad_edge_weight};
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
        auto result = EdgeOperation::apply(edge_index, edge_vec, edge_weight, grad_edge_vec,
                                              grad_edge_weight);
        auto grad_positions = result[0];
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
