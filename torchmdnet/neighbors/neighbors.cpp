#include <torch/extension.h>

TORCH_LIBRARY(torchmdnet_neighbors, m) {
    m.def("get_neighbor_pairs(str strategy, Tensor positions, Tensor batch, Tensor box_vectors, bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs, Tensor num_pairs)");
}
