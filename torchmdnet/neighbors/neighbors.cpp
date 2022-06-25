#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_list(Tensor positions, Tensor batch, float radius, int max_hash_size) -> (Tensor rows, Tensor columns, Tensor distance)");
}
