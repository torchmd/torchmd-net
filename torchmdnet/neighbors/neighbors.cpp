#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_pairs(Tensor positions, Scalar cutoff, Scalar max_num_neighbors) -> (Tensor neighbors, Tensor distances)");
}