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
using torch::indexing::None;
using torch::indexing::Slice;

static tuple<Tensor, Tensor, Tensor, Tensor>
forward(const Tensor& positions, const Tensor& batch, const Tensor& in_box_vectors,
        bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
        const Scalar& max_num_pairs, bool loop, bool include_transpose) {
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
	// Check that the box is a valid triclinic box, in the case of a box per sample we only check the first one
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
    return {neighbors, deltas, distances, num_pairs_found};
}

TORCH_LIBRARY_IMPL(torchmdnet_extensions, CPU, m) {
    m.impl("get_neighbor_pairs",
           [](const std::string& strategy, const Tensor& positions, const Tensor& batch,
              const Tensor& box_vectors, bool use_periodic, const Scalar& cutoff_lower,
              const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
              bool include_transpose) {
               return forward(positions, batch, box_vectors, use_periodic, cutoff_lower,
                              cutoff_upper, max_num_pairs, loop, include_transpose);
           });
}
