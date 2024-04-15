/* Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 *(See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 */
#include <torch/extension.h>
#include <tuple>

using std::tuple;
using torch::arange;
using torch::div;
using torch::frobenius_norm;
using torch::full;
using torch::hstack;
using torch::index_select;
using torch::kInt32;
using torch::outer;
using torch::round;
using torch::Scalar;
using torch::Tensor;
using torch::vstack;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;
using torch::indexing::None;
using torch::indexing::Slice;

static tuple<Tensor, Tensor, Tensor, Tensor>
forward_impl(const std::string& strategy, const Tensor& positions, const Tensor& batch,
             const Tensor& in_box_vectors, bool use_periodic, const Scalar& cutoff_lower,
             const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
             bool include_transpose) {
    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0,
                "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");
    TORCH_CHECK(cutoff_upper.to<double>() > 0, "Expected \"cutoff\" to be positive");
    Tensor box_vectors = in_box_vectors;
    const int n_batch = batch.max().item<int>() + 1;
    if (use_periodic) {
        if (box_vectors.dim() == 2) {
            box_vectors = box_vectors.unsqueeze(0).expand({n_batch, 3, 3});
        }
        TORCH_CHECK(box_vectors.dim() == 3, "Expected \"box_vectors\" to have two dimensions");
        TORCH_CHECK(box_vectors.size(1) == 3 && box_vectors.size(2) == 3,
                    "Expected \"box_vectors\" to have shape (n_batch, 3, 3)");
        // Ensure the box first dimension has size max(batch) + 1
        TORCH_CHECK(box_vectors.size(0) == n_batch,
                    "Expected \"box_vectors\" to have shape (n_batch, 3, 3)");
        // Check that the box is a valid triclinic box, in the case of a box per sample we only
        // check the first one
        double v[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                v[i][j] = box_vectors[0][i][j].item<double>();
        double c = cutoff_upper.to<double>();
        TORCH_CHECK(v[0][1] == 0, "Invalid box vectors: box_vectors[0][1] != 0");
        TORCH_CHECK(v[0][2] == 0, "Invalid box vectors: box_vectors[0][2] != 0");
        TORCH_CHECK(v[1][2] == 0, "Invalid box vectors: box_vectors[1][2] != 0");
        TORCH_CHECK(v[0][0] >= 2 * c, "Invalid box vectors: box_vectors[0][0] < 2*cutoff");
        TORCH_CHECK(v[1][1] >= 2 * c, "Invalid box vectors: box_vectors[1][1] < 2*cutoff");
        TORCH_CHECK(v[2][2] >= 2 * c, "Invalid box vectors: box_vectors[2][2] < 2*cutoff");
        TORCH_CHECK(v[0][0] >= 2 * v[1][0],
                    "Invalid box vectors: box_vectors[0][0] < 2*box_vectors[1][0]");
        TORCH_CHECK(v[0][0] >= 2 * v[2][0],
                    "Invalid box vectors: box_vectors[0][0] < 2*box_vectors[1][0]");
        TORCH_CHECK(v[1][1] >= 2 * v[2][1],
                    "Invalid box vectors: box_vectors[1][1] < 2*box_vectors[2][1]");
    }
    TORCH_CHECK(max_num_pairs.toLong() > 0, "Expected \"max_num_neighbors\" to be positive");
    const int n_atoms = positions.size(0);
    Tensor neighbors = torch::empty({0}, positions.options().dtype(kInt32));
    Tensor distances = torch::empty({0}, positions.options());
    Tensor deltas = torch::empty({0}, positions.options());
    neighbors = torch::vstack((torch::tril_indices(n_atoms, n_atoms, -1, neighbors.options())));
    auto mask = index_select(batch, 0, neighbors.index({0, Slice()})) ==
                index_select(batch, 0, neighbors.index({1, Slice()}));
    neighbors = neighbors.index({Slice(), mask}).to(kInt32);
    deltas = index_select(positions, 0, neighbors.index({0, Slice()})) -
             index_select(positions, 0, neighbors.index({1, Slice()}));
    if (use_periodic) {
        const auto pair_batch = batch.index({neighbors.index({0, Slice()})});
        const auto scale3 =
            round(deltas.index({Slice(), 2}) / box_vectors.index({pair_batch, 2, 2}));
        deltas.index_put_({Slice(), 0}, deltas.index({Slice(), 0}) -
                                            scale3 * box_vectors.index({pair_batch, 2, 0}));
        deltas.index_put_({Slice(), 1}, deltas.index({Slice(), 1}) -
                                            scale3 * box_vectors.index({pair_batch, 2, 1}));
        deltas.index_put_({Slice(), 2}, deltas.index({Slice(), 2}) -
                                            scale3 * box_vectors.index({pair_batch, 2, 2}));
        const auto scale2 =
            round(deltas.index({Slice(), 1}) / box_vectors.index({pair_batch, 1, 1}));
        deltas.index_put_({Slice(), 0}, deltas.index({Slice(), 0}) -
                                            scale2 * box_vectors.index({pair_batch, 1, 0}));
        deltas.index_put_({Slice(), 1}, deltas.index({Slice(), 1}) -
                                            scale2 * box_vectors.index({pair_batch, 1, 1}));
        const auto scale1 =
            round(deltas.index({Slice(), 0}) / box_vectors.index({pair_batch, 0, 0}));
        deltas.index_put_({Slice(), 0}, deltas.index({Slice(), 0}) -
                                            scale1 * box_vectors.index({pair_batch, 0, 0}));
    }
    distances = frobenius_norm(deltas, 1);
    mask = (distances < cutoff_upper) * (distances >= cutoff_lower);
    neighbors = neighbors.index({Slice(), mask});
    deltas = deltas.index({mask, Slice()});
    distances = distances.index({mask});
    if (include_transpose) {
        neighbors = torch::hstack({neighbors, torch::stack({neighbors[1], neighbors[0]})});
        distances = torch::hstack({distances, distances});
        deltas = torch::vstack({deltas, -deltas});
    }
    if (loop) {
        const Tensor range = torch::arange(0, n_atoms, torch::kInt32);
        neighbors = torch::hstack({neighbors, torch::stack({range, range})});
        distances = torch::hstack({distances, torch::zeros_like(range)});
        deltas = torch::vstack({deltas, torch::zeros({n_atoms, 3}, deltas.options())});
    }
    Tensor num_pairs_found = torch::empty(1, distances.options().dtype(kInt32));
    num_pairs_found[0] = distances.size(0);
    // This seems wasteful, but it allows to enable torch.compile by guaranteeing that the output of
    // this operator has a predictable size Resize to max_num_pairs, add zeros if necessary
    int64_t extension = std::max(max_num_pairs.toLong() - distances.size(0), (int64_t)0);
    if (extension > 0) {
        deltas = torch::vstack({deltas, torch::zeros({extension, 3}, deltas.options())});
        distances = torch::hstack({distances, torch::zeros({extension}, distances.options())});
        // For the neighbors add (-1,-1) pairs to fill the tensor
        neighbors = torch::hstack(
            {neighbors, torch::full({2, extension}, -1, neighbors.options().dtype(kInt32))});
    }
    return {neighbors, deltas, distances, num_pairs_found};
}

// The backwards operation is implemented fully in pytorch so that it can be differentiated a second
// time automatically via Autograd.
static Tensor backward_impl(const Tensor& grad_edge_vec, const Tensor& grad_edge_weight,
                            const Tensor& edge_index, const Tensor& edge_vec,
                            const Tensor& edge_weight, const int64_t num_atoms) {
    auto zero_mask = edge_weight.eq(0);
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
    return grad_positions;
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
        static auto fwd =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("torchmdnet_extensions::get_neighbor_pairs_fwd", "")
                .typed<decltype(forward_impl)>();
        Tensor neighbors, deltas, distances, i_curr_pair;
        std::tie(neighbors, deltas, distances, i_curr_pair) =
            fwd.call(strategy, positions, batch, box_vectors, use_periodic, cutoff_lower,
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
        static auto backward =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("torchmdnet_extensions::get_neighbor_pairs_bkwd", "")
                .typed<decltype(backward_impl)>();
        auto grad_positions = backward.call(grad_edge_vec, grad_edge_weight, edge_index, edge_vec,
                                            edge_weight, num_atoms);
        Tensor ignore;
        return {ignore, grad_positions, ignore, ignore, ignore, ignore,
                ignore, ignore,         ignore, ignore, ignore};
    }
};

// By registering as a CompositeImplicitAutograd we can torch.compile this Autograd function.
// This mode will generate meta registration for NeighborAutograd automatically, in this case using
// the
//  meta registrations provided for the forward and backward functions python side.
// We provide meta registrations python side because it is the recommended way to do it.
TORCH_LIBRARY_IMPL(torchmdnet_extensions, CompositeImplicitAutograd, m) {
    m.impl("get_neighbor_pairs",
           [](const std::string& strategy, const Tensor& positions, const Tensor& batch,
              const Tensor& box_vectors, bool use_periodic, const Scalar& cutoff_lower,
              const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
              bool include_transpose) {
               auto result = NeighborAutograd::apply(strategy, positions, batch, box_vectors,
                                                     use_periodic, cutoff_lower, cutoff_upper,
                                                     max_num_pairs, loop, include_transpose);
               return std::make_tuple(result[0], result[1], result[2], result[3]);
           });
}

// The registration for the CUDA version of the forward function is done in a separate .cu file.
TORCH_LIBRARY_IMPL(torchmdnet_extensions, CPU, m) {
    m.impl("get_neighbor_pairs_fwd", forward_impl);
}

// Ideally we would register this just once using CompositeImplicitAutograd, but this causes a
// segfault
//  when trying to torch.compile this function.
// Doing it this way prints a message about Autograd not being provided a backward function of the
// backward function. It gets it from the implementation just fine now, but warns that in the future
// this will be deprecated.
TORCH_LIBRARY_IMPL(torchmdnet_extensions, CPU, m) {
    m.impl("get_neighbor_pairs_bkwd", backward_impl);
}

TORCH_LIBRARY_IMPL(torchmdnet_extensions, CUDA, m) {
    m.impl("get_neighbor_pairs_bkwd", backward_impl);
}
