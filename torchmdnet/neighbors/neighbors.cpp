#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_pairs(Tensor positions) -> (Tensor neighbors, Tensor distances)");
}