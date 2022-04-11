#include <torch/extension.h>
#include <tuple>

using std::tuple;
using torch::div;
using torch::index_select;
using torch::arange;
using torch::frobenius_norm;
using torch::kInt32;
using torch::Scalar;
using torch::stack;
using torch::Tensor;

static tuple<Tensor, Tensor> forward(const Tensor& positions,
                                     const Scalar& cutoff,
                                     const Scalar& max_num_neighbors) {

    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0, "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    const int num_atoms = positions.size(0);
    const int num_pairs = num_atoms * (num_atoms - 1) / 2;

    const Tensor indices = arange(0, num_pairs, positions.options().dtype(kInt32));
    Tensor rows = (((8 * indices + 1).sqrt() + 1) / 2).floor().to(kInt32);
    rows -= (rows * (rows - 1) > 2 * indices).to(kInt32);
    const Tensor columns = indices - div(rows * (rows - 1), 2, "floor");

    const Tensor neighbors = stack({rows, columns});
    const Tensor vectors = index_select(positions, 0, rows) - index_select(positions, 0, columns);
    const Tensor distances = frobenius_norm(vectors, 1);

    return {neighbors, distances};
}

TORCH_LIBRARY_IMPL(neighbors, CPU, m) {
    m.impl("get_neighbor_pairs", &forward);
}