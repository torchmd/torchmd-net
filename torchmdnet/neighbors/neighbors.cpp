#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_list(Tensor positions) -> (Tensor rows, Tensor columns, Tensor distance)");
}