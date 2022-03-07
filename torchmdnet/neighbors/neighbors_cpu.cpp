#include <torch/extension.h>
#include <tuple>

using std::tuple;
using torch::index_select;
using torch::arange;
using torch::div;
using torch::frobenius_norm;
using torch::remainder;
using torch::Tensor;

static tuple<Tensor, Tensor, Tensor> forward(const Tensor& positions) {

    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    const int num_atoms = positions.size(0);
    const int num_neighbors = num_atoms * (num_atoms - 1);

    const Tensor indices = arange(0, num_neighbors, positions.options().dtype(torch::kInt32));
    const Tensor rows = div(indices, num_atoms - 1, "floor");
    const Tensor columns = remainder(indices + div(indices, num_atoms, "floor") + 1, num_atoms);

    const Tensor vectors = index_select(positions, 0, rows) - index_select(positions, 0, columns);
    const Tensor distances = frobenius_norm(vectors, 1);

    return {rows, columns, distances};
}

TORCH_LIBRARY_IMPL(neighbors, CPU, m) {
    m.impl("get_neighbor_list", &forward);
}