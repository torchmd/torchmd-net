/* Raul P. Pelaez 2023. Batched cell list neighbor list implementation for CUDA.

 */
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <torch/extension.h>
#include <tuple>
using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using std::make_tuple;
using std::max;
using torch::empty;
using torch::full;
using torch::kInt32;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;
using torch::zeros;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <typename scalar_t> struct scalar3 {
    scalar_t x, y, z;
};

template <typename scalar_t, int num_dims>
using Accessor = PackedTensorAccessor32<scalar_t, num_dims, RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, RestrictPtrTraits>();
};

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x){};
template <> __device__ __forceinline__ float sqrt_(float x) {
    return ::sqrtf(x);
};
template <> __device__ __forceinline__ double sqrt_(double x) {
    return ::sqrt(x);
};

template <typename scalar_t>
__global__ void
backward_kernel(const Accessor<int32_t, 2> neighbors, const Accessor<scalar_t, 2> deltas,
                const Accessor<scalar_t, 1> distances, const Accessor<scalar_t, 1> grad_distances,
                Accessor<scalar_t, 2> grad_positions) {
    // What the backward kernel does:
    // For each pair of atoms, it calculates the gradient of the distance between them
    // with respect to the positions of the atoms.
    // The gradient is then added to the gradient of the positions.
    // The gradient of the distance is calculated using the chain rule:
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs)
        return;

    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    if (i_atom < 0)
        return;

    const int32_t i_comp = blockIdx.z;
    const scalar_t grad = deltas[i_pair][i_comp] / distances[i_pair] * grad_distances[i_pair];
    atomicAdd(&grad_positions[i_atom][i_comp], (i_dir ? -1 : 1) * grad);
}

static void checkInput(const Tensor& positions, const Tensor& batch) {
    // Batch contains the molecule index for each atom in positions
    // Neighbors are only calculated within the same molecule
    // Batch is a 1D tensor of size (N_atoms)
    // Batch is assumed to be sorted and starts at zero.
    // Batch is assumed to be contiguous
    // Batch is assumed to be of type torch::kLong
    // Batch is assumed to be non-negative
    // Each batch can have a different number of atoms
    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0,
                "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    TORCH_CHECK(batch.dim() == 1, "Expected \"batch\" to have one dimension");
    TORCH_CHECK(batch.size(0) == positions.size(0),
                "Expected the 1st dimension size of \"batch\" to be the same as the 1st dimension "
                "size of \"positions\"");
    TORCH_CHECK(batch.is_contiguous(), "Expected \"batch\" to be contiguous");
    TORCH_CHECK(batch.dtype() == torch::kInt64, "Expected \"batch\" to be of type torch::kLong");
}

/*
 * @brief Encodes an unsigned integer lower than 1024 as a 32 bit integer by filling every third
 * bit.
 * @param i The integer to encode
 * @return The encoded integer
 */
inline __host__ __device__ uint encodeMorton(const uint& i) {
    uint x = i;
    x &= 0x3ff;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
}

/*
 * @brief Interleave three 10 bit numbers in 32 bits, producing a Z order Morton hash
 * @param ci The cell index
 * @return The Morton hash
 */
inline __host__ __device__ uint hashMorton(int3 ci) {
    return encodeMorton(ci.x) | (encodeMorton(ci.y) << 1) | (encodeMorton(ci.z) << 2);
}

/*
 * @brief Takes a point to the unit cell in the range [-0.5, 0.5]*box_size using Minimum Image
 * Convention
 * @param p The point position
 * @param box_size The box size
 * @return The point in the unit cell
 */
template <typename scalar_t>
__device__ auto takeToUnitCell(scalar3<scalar_t> p, scalar3<scalar_t> box_size) {
    p.x = p.x - floorf(p.x / box_size.x + scalar_t(0.5)) * box_size.x;
    p.y = p.y - floorf(p.y / box_size.y + scalar_t(0.5)) * box_size.y;
    p.z = p.z - floorf(p.z / box_size.z + scalar_t(0.5)) * box_size.z;
    return p;
}

/*
 * @brief Calculates the cell dimensions for a given box size and cutoff
 * @param box_size The box size
 * @param cutoff The cutoff
 * @return The cell dimensions
 */
template <typename scalar_t>
__host__ __device__ int3 getCellDimensions(scalar3<scalar_t> box_size, scalar_t cutoff) {
    int3 cell_dim = make_int3(box_size.x / cutoff, box_size.y / cutoff, box_size.z / cutoff);
    // Minimum 3 cells in each dimension
    cell_dim.x = thrust::max(cell_dim.x, 3);
    cell_dim.y = thrust::max(cell_dim.y, 3);
    cell_dim.z = thrust::max(cell_dim.z, 3);
// In the host, throw if there are more than 1024 cells in any dimension
#ifndef __CUDA_ARCH__
    if (cell_dim.x > 1024 || cell_dim.y > 1024 || cell_dim.z > 1024) {
        throw std::runtime_error("Too many cells in one dimension. Maximum is 1024");
    }
#endif
    return cell_dim;
}

/*
 * @brief Get the cell index of a point
 * @param p The point position
 * @param box_size The size of the box in each dimension
 * @param cutoff The cutoff
 * @return The cell index
 */
template <typename scalar_t>
__device__ int3 getCell(scalar3<scalar_t> p, scalar3<scalar_t> box_size, scalar_t cutoff) {
    p = takeToUnitCell(p, box_size);
    // Take to the [0, box_size] range and divide by cutoff (which is the cell size)
    int cx = floorf((p.x + scalar_t(0.5) * box_size.x) / cutoff);
    int cy = floorf((p.y + scalar_t(0.5) * box_size.y) / cutoff);
    int cz = floorf((p.z + scalar_t(0.5) * box_size.z) / cutoff);
    int3 cell_dim = getCellDimensions(box_size, cutoff);
    // Wrap around. If the position of a particle is exactly box_size, it will be in the last cell,
    // which results in an illegal access down the line.
    if (cx == cell_dim.x)
        cx = 0;
    if (cy == cell_dim.y)
        cy = 0;
    if (cz == cell_dim.z)
        cz = 0;
    return make_int3(cx, cy, cz);
}

/*
 * @brief Get the index of a cell in a 1D array of cells.
 * @param cell The cell coordinates, assumed to be in the range [0, cell_dim].
 * @param cell_dim The number of cells in each dimension
 */
__device__ int getCellIndex(int3 cell, int3 cell_dim) {
    return cell.x + cell_dim.x * (cell.y + cell_dim.y * cell.z);
}

/*
  @brief Fold a cell coordinate to the range [0, cell_dim)
  @param cell The cell coordinate
  @param cell_dim The dimensions of the grid
  @return The folded cell coordinate
*/
__device__ int3 getPeriodicCell(int3 cell, int3 cell_dim) {
    int3 periodic_cell = cell;
    if (cell.x < 0)
        periodic_cell.x += cell_dim.x;
    if (cell.x >= cell_dim.x)
        periodic_cell.x -= cell_dim.x;
    if (cell.y < 0)
        periodic_cell.y += cell_dim.y;
    if (cell.y >= cell_dim.y)
        periodic_cell.y -= cell_dim.y;
    if (cell.z < 0)
        periodic_cell.z += cell_dim.z;
    if (cell.z >= cell_dim.z)
        periodic_cell.z -= cell_dim.z;
    return periodic_cell;
}

// Assign a hash to each atom based on its position and batch.
// This hash is such that atoms in the same cell and batch have the same hash.
template <typename scalar_t>
__global__ void assignHash(const Accessor<scalar_t, 2> positions, uint64_t* hash_keys,
                           Accessor<int32_t, 1> hash_values, const Accessor<int64_t, 1> batch,
                           scalar3<scalar_t> box_size, scalar_t cutoff, int32_t num_atoms) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    const int32_t i_batch = batch[i_atom];
    // Move to the unit cell
    scalar3<scalar_t> pi = {positions[i_atom][0], positions[i_atom][1], positions[i_atom][2]};
    auto ci = getCell(pi, box_size, cutoff);
    // Calculate the hash
    const int32_t hash = hashMorton(ci);
    // Create a hash combining the Morton hash and the batch index, so that atoms in the same cell
    // are contiguous
    const int64_t hash_final = (static_cast<int64_t>(hash) << 32) | i_batch;
    hash_keys[i_atom] = hash_final;
    hash_values[i_atom] = i_atom;
}

// Adaptor from pytorch cached allocator to thrust
template <typename T> class CudaAllocator {
public:
    using value_type = T;
    CudaAllocator() {
    }
    T* allocate(std::ptrdiff_t num_elements) {
        return static_cast<T*>(
            at::cuda::getCUDADeviceAllocator()->raw_allocate(num_elements * sizeof(T)));
    }
    void deallocate(T* ptr, size_t) {
        at::cuda::getCUDADeviceAllocator()->raw_deallocate(ptr);
    }
};

/*
 * @brief Sort the positions by hash, based on the cell assigned to each position and the batch
 * index
 * @param positions The positions of the atoms
 * @param batch The batch index of each atom
 * @param box_size The size of the box in each dimension
 * @param cutoff The cutoff
 * @return A tuple of the sorted positions and the original indices of each atom in the sorted list
 */

static auto sortPositionsByHash(const Tensor& positions, const Tensor& batch,
                                const Tensor& box_size, const Scalar& cutoff) {

    const int num_atoms = positions.size(0);
    const auto options = positions.options();
    thrust::device_vector<uint64_t> hash_keys(num_atoms);
    Tensor hash_values = empty({num_atoms}, options.dtype(torch::kInt32));
    const int threads = 128;
    const int blocks = (num_atoms + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "assignHash", [&] {
        scalar_t cutoff_ = cutoff.to<scalar_t>();
        scalar3<scalar_t> box_size_ = {box_size[0].item<scalar_t>(), box_size[1].item<scalar_t>(),
                                       box_size[2].item<scalar_t>()};
        assignHash<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(positions), thrust::raw_pointer_cast(hash_keys.data()),
            get_accessor<int32_t, 1>(hash_values), get_accessor<int64_t, 1>(batch), box_size_,
            cutoff_, num_atoms);
    });
    thrust::device_ptr<int32_t> index_ptr(hash_values.data_ptr<int32_t>());
    CudaAllocator<char> allocator;
    thrust::sort_by_key(thrust::cuda::par.on(stream), hash_keys.begin(), hash_keys.end(),
                        index_ptr);
    Tensor sorted_positions = positions.index_select(0, hash_values);
    return std::make_tuple(sorted_positions, hash_values);
}

template <typename scalar_t>
__global__ void fillCellOffsetsD(const Accessor<scalar_t, 2> sorted_positions,
                                 const Accessor<int32_t, 1> sorted_indices,
                                 Accessor<int32_t, 1> cell_start, Accessor<int32_t, 1> cell_end,
                                 const Accessor<int64_t, 1> batch, scalar3<scalar_t> box_size,
                                 scalar_t cutoff) {
    // Since positions are sorted by cell, for a given atom, if the previous atom is in a different
    // cell, then the current atom is the first atom in its cell We use this fact to fill the
    // cell_start and cell_end arrays
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= sorted_positions.size(0))
        return;
    const scalar3<scalar_t> pi = {sorted_positions[i_atom][0], sorted_positions[i_atom][1],
                                  sorted_positions[i_atom][2]};
    const int3 cell_dim = getCellDimensions(box_size, cutoff);
    const int icell = getCellIndex(getCell(pi, box_size, cutoff), cell_dim);
    int im1_cell;
    if (i_atom > 0) {
        int im1 = i_atom - 1;
        const scalar3<scalar_t> pim1 = {sorted_positions[im1][0], sorted_positions[im1][1],
                                        sorted_positions[im1][2]};
        im1_cell = getCellIndex(getCell(pim1, box_size, cutoff), cell_dim);
    } else {
        im1_cell = 0;
    }
    if (icell != im1_cell || i_atom == 0) {
        int n_cells = cell_start.size(0);
        cell_start[icell] = i_atom;
        if (i_atom > 0)
            cell_end[im1_cell] = i_atom;
    }
    if (i_atom == sorted_positions.size(0) - 1) {
        cell_end[icell] = i_atom + 1;
    }
}

/*
  @brief
  Fill the cell offsets for each batch, identifying the start and end of each cell in the sorted
  positions
  @param sorted_positions The positions sorted by cell
  @param sorted_indices The original indices of the sorted positions
  @param batch The batch index of each position
  @param box_size The size of the box
  @param cutoff The cutoff distance
  @return A tuple of cell_start and cell_end arrays
*/
static auto fillCellOffsets(const Tensor& sorted_positions, const Tensor& sorted_indices,
                            const Tensor& batch, const Tensor& box_size, const Scalar& cutoff) {
    const TensorOptions options = sorted_positions.options();

    int3 cell_dim;
    AT_DISPATCH_FLOATING_TYPES(sorted_positions.scalar_type(), "fillCellOffsets", [&] {
        scalar_t cutoff_ = cutoff.to<scalar_t>();
        scalar3<scalar_t> box_size_ = {box_size[0].item<scalar_t>(), box_size[1].item<scalar_t>(),
                                       box_size[2].item<scalar_t>()};
        cell_dim = getCellDimensions(box_size_, cutoff_);
    });
    const int num_cells = cell_dim.x * cell_dim.y * cell_dim.z;
    const Tensor cell_start = full({num_cells}, -1, options.dtype(torch::kInt));
    const Tensor cell_end = empty({num_cells}, options.dtype(torch::kInt));
    const int threads = 128;
    const int blocks = (sorted_positions.size(0) + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(sorted_positions.scalar_type(), "fillCellOffsets", [&] {
        auto stream = at::cuda::getCurrentCUDAStream();
        scalar_t cutoff_ = cutoff.to<scalar_t>();
        scalar3<scalar_t> box_size_ = {box_size[0].item<scalar_t>(), box_size[1].item<scalar_t>(),
                                       box_size[2].item<scalar_t>()};
        fillCellOffsetsD<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(sorted_positions), get_accessor<int32_t, 1>(sorted_indices),
            get_accessor<int32_t, 1>(cell_start), get_accessor<int32_t, 1>(cell_end),
            get_accessor<int64_t, 1>(batch), box_size_, cutoff_);
    });
    return std::make_tuple(cell_start, cell_end);
}

/*
  @brief Get the cell index of the i'th neighboring cell for a given cell
  @param cell_i The cell coordinates
  @param i The index of the neighboring cell, from 0 to 26
  @param cell_dim The dimensions of the cell grid
  @return The cell index of the i'th neighboring cell
*/
__device__ int getNeighborCellIndex(int3 cell_i, int i, int3 cell_dim) {
    auto cell_j = cell_i;
    cell_j.x += i % 3 - 1;
    cell_j.y += (i / 3) % 3 - 1;
    cell_j.z += i / 9 - 1;
    cell_j = getPeriodicCell(cell_j, cell_dim);
    int icellj = getCellIndex(cell_j, cell_dim);
    return icellj;
}

// Traverse the cell list for each atom and find the neighbors
template <typename scalar_t>
__global__ void
forward_kernel(const Accessor<scalar_t, 2> sorted_positions,
               const Accessor<int32_t, 1> original_index, const Accessor<int64_t, 1> batch,
               const Accessor<int32_t, 1> cell_start, const Accessor<int32_t, 1> cell_end,
               Accessor<int32_t, 2> neighbors, Accessor<scalar_t, 2> deltas,
               Accessor<scalar_t, 1> distances, Accessor<int32_t, 1> i_curr_pair, int num_atoms,
               int num_pairs, scalar3<scalar_t> box_size, scalar_t cutoff_lower,
               scalar_t cutoff_upper, bool loop, bool include_transpose) {
    // Each atom traverses the cells around it and finds the neighbors
    // Atoms for all batches are placed in the same cell list, but other batches are ignored while
    // traversing
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    const int ori = original_index[i_atom];
    const auto i_batch = batch[ori];
    const scalar3<scalar_t> pi = {sorted_positions[i_atom][0], sorted_positions[i_atom][1],
                                  sorted_positions[i_atom][2]};
    const int3 cell_i = getCell(pi, box_size, cutoff_upper);
    const int3 cell_dim = getCellDimensions(box_size, cutoff_upper);
    // Loop over the 27 cells around the current cell
    for (int i = 0; i < 27; i++) {
        int icellj = getNeighborCellIndex(cell_i, i, cell_dim);
        const int firstParticle = cell_start[icellj];
        if (firstParticle != -1) { // Continue only if there are particles in this cell
            // Index of the last particle in the cell's list
            const int lastParticle = cell_end[icellj];
            const int nincell = lastParticle - firstParticle;
            for (int j = 0; j < nincell; j++) {
                const int cur_j = j + firstParticle;
                const int orj = original_index[cur_j];
                const auto j_batch = batch[orj];
                if (j_batch >
                    i_batch) // Particles are sorted by batch after cell, so we can break early here
                    break;
                const bool includePair =
                    (j_batch == i_batch) and
                    ((orj != ori and (orj < ori or include_transpose)) or (loop and orj == ori));
                if (includePair) {
                    const scalar3<scalar_t> pj = {sorted_positions[cur_j][0],
                                                  sorted_positions[cur_j][1],
                                                  sorted_positions[cur_j][2]};
                    const scalar_t dx = pi.x - pj.x;
                    const scalar_t dy = pi.y - pj.y;
                    const scalar_t dz = pi.z - pj.z;
                    const scalar_t distance2 = dx * dx + dy * dy + dz * dz;
                    const scalar_t cutoff_upper2 = cutoff_upper * cutoff_upper;
                    const scalar_t cutoff_lower2 = cutoff_lower * cutoff_lower;
                    if ((distance2 <= cutoff_upper2 and distance2 >= cutoff_lower2) or
                        (loop and orj == ori)) {
                        const int32_t i_pair = atomicAdd(&i_curr_pair[0], 1);
                        // We handle too many neighbors outside of the kernel
                        if (i_pair < neighbors.size(1)) {
                            neighbors[0][i_pair] = ori;
                            neighbors[1][i_pair] = orj;
                            deltas[i_pair][0] = dx;
                            deltas[i_pair][1] = dy;
                            deltas[i_pair][2] = dz;
                            distances[i_pair] = sqrt_(distance2);
                        } // endif
                    }     // endif
                }         // endfor
            }             // endif
        }                 // endfor
    }                     // endfor
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& positions, const Tensor& batch,
                               const Tensor& box_size, const Scalar& cutoff_lower,
                               const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
                               bool include_transpose, bool checkErrors) {
        // The algorithm for the cell list construction can be summarized in three separate steps:
        //         1. Hash (label) the particles according to the cell (bin) they lie in.
        //         2. Sort the particles and hashes using the hashes as the ordering label
        //         (technically this is known as sorting by key). So that particles with positions
        //         lying in the same cell become contiguous in memory.
        //         3. Identify where each cell starts and ends in the sorted particle positions
        //         array.
        checkInput(positions, batch);
        TORCH_CHECK(box_size.size(0) == 3, "Expected \"box_size\" to have 3 elements");
        const auto max_num_pairs_ = max_num_pairs.toLong();
        TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
        const int num_atoms = positions.size(0);
        const int num_pairs = max_num_pairs_;
        const TensorOptions options = positions.options();
        // Steps 1 and 2
        Tensor sorted_positions, hash_values;
        std::tie(sorted_positions, hash_values) =
            sortPositionsByHash(positions, batch, box_size, cutoff_upper);
        Tensor cell_start, cell_end;
        std::tie(cell_start, cell_end) =
            fillCellOffsets(sorted_positions, hash_values, batch, box_size, cutoff_upper);
        const Tensor neighbors = full({2, num_pairs}, -1, options.dtype(kInt32));
        const Tensor deltas = empty({num_pairs, 3}, options);
        const Tensor distances = full(num_pairs, 0, options);
        const Tensor i_curr_pair = zeros(1, options.dtype(kInt32));
        const auto stream = getCurrentCUDAStream(positions.get_device());
        { // Use the cell list for each batch to find the neighbors
            const CUDAStreamGuard guard(stream);
            AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "forward", [&] {
                const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
                TORCH_CHECK(cutoff_upper_ > 0, "Expected cutoff_upper to be positive");
                const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
                const scalar3<scalar_t> box_size_ = {box_size[0].item<scalar_t>(),
                                                     box_size[1].item<scalar_t>(),
                                                     box_size[2].item<scalar_t>()};
                const int threads = 128;
                const int blocks = (num_atoms + threads - 1) / threads;
                forward_kernel<<<blocks, threads, 0, stream>>>(
                    get_accessor<scalar_t, 2>(sorted_positions),
                    get_accessor<int32_t, 1>(hash_values), get_accessor<int64_t, 1>(batch),
                    get_accessor<int32_t, 1>(cell_start), get_accessor<int32_t, 1>(cell_end),
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 1>(distances), get_accessor<int32_t, 1>(i_curr_pair),
                    num_atoms, num_pairs, box_size_, cutoff_lower_, cutoff_upper_, loop,
                    include_transpose);
            });
        }
        // Synchronize and check the number of pairs found. Note that this is incompatible with CUDA
        // graphs
        if (checkErrors) {
            int num_found_pairs = i_curr_pair[0].item<int32_t>();
            TORCH_CHECK(num_found_pairs <= max_num_pairs_,
                        "Too many neighbor pairs found. Maximum is " +
                            std::to_string(max_num_pairs_),
                        " but found " + std::to_string(num_found_pairs));
        }
        ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;
        return {neighbors, deltas, distances};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {
        const Tensor grad_distances = grad_inputs[1];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks_x = max((num_pairs + num_threads - 1) / num_threads, 1);
        const dim3 blocks(num_blocks_x, 2, 3);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        const tensor_list data = ctx->get_saved_variables();
        const Tensor neighbors = data[0];
        const Tensor deltas = data[1];
        const Tensor distances = data[2];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(
            grad_distances.scalar_type(), "get_neighbor_pairs_backward", [&]() {
                const CUDAStreamGuard guard(stream);
                backward_kernel<<<blocks, num_threads, 0, stream>>>(
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 1>(distances), get_accessor<scalar_t, 1>(grad_distances),
                    get_accessor<scalar_t, 2>(grad_positions));
            });

        return {grad_positions, Tensor(), Tensor()};
    }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs_cell",
           [](const Tensor& positions, const Tensor& batch, const Tensor& box_size,
              const Scalar& cutoff_lower, const Scalar& cutoff_upper, const Scalar& max_num_pairs,
              bool loop, bool include_transpose, bool checkErrors) {
               const tensor_list results =
                   Autograd::apply(positions, batch, box_size, cutoff_lower, cutoff_upper,
                                   max_num_pairs, loop, include_transpose, checkErrors);
               return std::make_tuple(results[0], results[1], results[2]);
           });
}
