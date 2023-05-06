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
using torch::indexing::Slice;

static tuple<Tensor, Tensor, Tensor, Tensor>
forward(const Tensor& positions, const Tensor& batch, const Tensor& box_vectors, bool use_periodic,
        const Scalar& cutoff_lower, const Scalar& cutoff_upper, const Scalar& max_num_pairs,
        bool loop, bool include_transpose) {
    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0,
                "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    TORCH_CHECK(cutoff_upper.to<double>() > 0, "Expected \"cutoff\" to be positive");
    if (use_periodic) {
        TORCH_CHECK(box_vectors.dim() == 2, "Expected \"box_vectors\" to have two dimensions");
        TORCH_CHECK(box_vectors.size(0) == 3 && box_vectors.size(1) == 3,
                    "Expected \"box_vectors\" to have shape (3, 3)");
        double v[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                v[i][j] = box_vectors[i][j].item<double>();
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
    const int n_batches = batch[n_atoms - 1].item<int>() + 1;
    int current_offset = 0;
    std::vector<int> batch_i;
    int n_pairs = 0;
    Tensor neighbors = torch::empty({0}, positions.options().dtype(kInt32));
    Tensor distances = torch::empty({0}, positions.options());
    Tensor deltas = torch::empty({0}, positions.options());
    for (int i = 0; i < n_batches; i++) {
        batch_i.clear();
        for (int j = current_offset; j < n_atoms; j++) {
            if (batch[j].item<int>() == i) {
                batch_i.push_back(j);
            } else {
                break;
            }
        }

        const int n_atoms_i = batch_i.size();
        Tensor positions_i = index_select(positions, 0, torch::tensor(batch_i, kInt32));
        Tensor indices_i =
            arange(0, n_atoms_i * (n_atoms_i - 1l) / 2l, positions.options().dtype(torch::kLong));
        Tensor rows_i = (((8l * indices_i + 1l).sqrt() + 1l) / 2l).floor().to(torch::kLong);
        rows_i -= (rows_i * (rows_i - 1l) > 2l * indices_i).to(torch::kLong);
        Tensor columns_i = indices_i - div(rows_i * (rows_i - 1l), 2, "floor");
        Tensor neighbors_i = vstack({rows_i, columns_i});
        Tensor deltas_i =
            index_select(positions_i, 0, rows_i) - index_select(positions_i, 0, columns_i);
        if (use_periodic) {
            deltas_i -= outer(round(deltas_i.index({Slice(), 2}) / box_vectors.index({2, 2})),
                              box_vectors.index({2}));
            deltas_i -= outer(round(deltas_i.index({Slice(), 1}) / box_vectors.index({1, 1})),
                              box_vectors.index({1}));
            deltas_i -= outer(round(deltas_i.index({Slice(), 0}) / box_vectors.index({0, 0})),
                              box_vectors.index({0}));
        }
        Tensor distances_i = frobenius_norm(deltas_i, 1);
        const Tensor mask_upper = distances_i < cutoff_upper;
        const Tensor mask_lower = distances_i >= cutoff_lower;
        const Tensor mask = mask_upper * mask_lower;
        neighbors_i = neighbors_i.index({Slice(), mask}) + current_offset;
        // Add the transposed pairs
        if (include_transpose) {
            neighbors_i =
                torch::hstack({neighbors_i, torch::stack({neighbors_i[1], neighbors_i[0]})});
        }
        // Add self interaction using batch_i
        if (loop) {
            const Tensor batch_i_tensor = torch::tensor(batch_i, kInt32);
            neighbors_i =
                torch::hstack({neighbors_i, torch::stack({batch_i_tensor, batch_i_tensor})});
        }
        n_pairs += neighbors_i.size(1);
        TORCH_CHECK(n_pairs >= 0,
                    "The maximum number of pairs has been exceed! Increase \"max_num_neighbors\"");
        neighbors = torch::hstack({neighbors, neighbors_i});
        current_offset += n_atoms_i;
    }
    if (n_batches > 1) {
        neighbors = torch::cat(neighbors, 0).to(kInt32);
    }
    deltas = index_select(positions, 0, neighbors[0]) - index_select(positions, 0, neighbors[1]);
    if (use_periodic) {
        deltas -= outer(round(deltas.index({Slice(), 2}) / box_vectors.index({2, 2})),
                        box_vectors.index({2}));
        deltas -= outer(round(deltas.index({Slice(), 1}) / box_vectors.index({1, 1})),
                        box_vectors.index({1}));
        deltas -= outer(round(deltas.index({Slice(), 0}) / box_vectors.index({0, 0})),
                        box_vectors.index({0}));
    }
    distances = frobenius_norm(deltas, 1);
    Tensor num_pairs_found = torch::empty(1, distances.options().dtype(kInt32));
    num_pairs_found[0] = distances.size(0);
    return {neighbors, deltas, distances, num_pairs_found};
}

TORCH_LIBRARY_IMPL(neighbors, CPU, m) {
    m.impl("get_neighbor_pairs_brute", &forward);
    m.impl("get_neighbor_pairs_shared", &forward);
    m.impl("get_neighbor_pairs_cell", &forward);
}
