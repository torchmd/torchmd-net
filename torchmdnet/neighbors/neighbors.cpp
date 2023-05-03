#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_pairs(Tensor positions, Tensor batch, Tensor box_size, Scalar cutoff, Scalar max_num_pairs, bool check_errors) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs)");
    m.def("get_neighbor_pairs_cell(Tensor positions, Tensor batch, Tensor box_size,Scalar cutoff, Scalar max_num_pairs, bool check_errors) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs)");
}