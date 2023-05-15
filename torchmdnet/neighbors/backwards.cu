/* Raul P. Pelaez 2023. Backwards pass for the CUDA neighbor list operation.
   Computes the gradient of the positions with respect to the distances and deltas.
 */
#include "common.cuh"

template <typename scalar_t>
__global__ void
backward_kernel(const Accessor<int32_t, 2> neighbors, const Accessor<scalar_t, 2> deltas,
                const Accessor<scalar_t, 2> grad_deltas, const Accessor<scalar_t, 1> distances,
                const Accessor<scalar_t, 1> grad_distances, Accessor<scalar_t, 2> grad_positions) {
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs)
        return;
    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    const int32_t i_comp = blockIdx.z;
    if (i_atom < 0) {
        return;
    }
    const scalar_t grad_deltas_ = grad_deltas[i_pair][i_comp];
    const scalar_t dist = distances[i_pair];
    const scalar_t grad_distances_ = deltas[i_pair][i_comp] / dist * grad_distances[i_pair];
    // Handle self interaction
    const scalar_t grad =
        (i_dir ? -1 : 1) *
        (i_atom == neighbors[1 - i_dir][i_pair] ? scalar_t(0.0) : (grad_deltas_ + grad_distances_));
    atomicAdd(&grad_positions[i_atom][i_comp], grad);
}


tensor_list common_backward(AutogradContext* ctx, tensor_list grad_inputs) {
    const Tensor grad_deltas = grad_inputs[1];
    const Tensor grad_distances = grad_inputs[2];
    const int num_atoms = ctx->saved_data["num_atoms"].toInt();
    const int num_pairs = grad_distances.size(0);
    const int num_threads = 128;
    const int num_blocks_x = std::max((num_pairs + num_threads - 1) / num_threads, 1);
    const dim3 blocks(num_blocks_x, 2, 3);
    const auto stream = getCurrentCUDAStream(grad_distances.get_device());

    const tensor_list data = ctx->get_saved_variables();
    const Tensor neighbors = data[0];
    const Tensor deltas = data[1];
    const Tensor distances = data[2];
    const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

    AT_DISPATCH_FLOATING_TYPES(grad_distances.scalar_type(), "getNeighborPairs::backward", [&]() {
        const CUDAStreamGuard guard(stream);
        backward_kernel<<<blocks, num_threads, 0, stream>>>(
            get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
            get_accessor<scalar_t, 2>(grad_deltas), get_accessor<scalar_t, 1>(distances),
            get_accessor<scalar_t, 1>(grad_distances), get_accessor<scalar_t, 2>(grad_positions));
    });

    return {grad_positions, Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(),       Tensor(), Tensor(), Tensor(), Tensor()};
}
