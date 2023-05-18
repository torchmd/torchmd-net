/* Raul P. Pelaez 2023
   Connection between the neighbor CUDA implementations and the torch extension.
   See neighbors.cpp for the definition of the torch extension functions.
 */
#include "neighbors_cuda_brute.cuh"
#include "neighbors_cuda_cell.cuh"
#include "neighbors_cuda_shared.cuh"

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs_brute",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
               tensor_list results;
               if (positions.size(0) >= 32768) {
                   // Revert to shared if there are too many particles, which brute can't handle
                   results = AutogradSharedCUDA::apply(positions, batch, cutoff_lower, cutoff_upper,
                                                       box_vectors, use_periodic, max_num_pairs,
                                                       loop, include_transpose);
               } else {
                   results = AutogradBruteCUDA::apply(positions, batch, cutoff_lower, cutoff_upper,
                                                      box_vectors, use_periodic, max_num_pairs,
                                                      loop, include_transpose);
               }
               return std::make_tuple(results[0], results[1], results[2], results[3]);
           });
    m.impl("get_neighbor_pairs_shared",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
               const tensor_list results = AutogradSharedCUDA::apply(
                   positions, batch, cutoff_lower, cutoff_upper, box_vectors, use_periodic,
                   max_num_pairs, loop, include_transpose);
               return std::make_tuple(results[0], results[1], results[2], results[3]);
           });
    m.impl("get_neighbor_pairs_cell",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_vectors,
              bool use_periodic, const Scalar& cutoff_lower, const Scalar& cutoff_upper,
              const Scalar& max_num_pairs, bool loop, bool include_transpose) {
               const tensor_list results = AutogradCellCUDA::apply(
                   positions, batch, box_vectors, use_periodic, cutoff_lower, cutoff_upper,
                   max_num_pairs, loop, include_transpose);
               return std::make_tuple(results[0], results[1], results[2], results[3]);
           });
}
