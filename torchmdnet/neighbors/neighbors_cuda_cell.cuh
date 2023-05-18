/* Raul P. Pelaez 2023. Batched cell list neighbor list implementation for CUDA.

 */
#ifndef NEIGHBOR_CUDA_CELL_H
#define NEIGHBOR_CUDA_CELL_H
#include "common.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

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
    p = rect::apply_pbc<scalar_t>(p, box_size);
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
    const uint32_t i_batch = batch[i_atom];
    // Move to the unit cell
    scalar3<scalar_t> pi = {positions[i_atom][0], positions[i_atom][1], positions[i_atom][2]};
    auto ci = getCell(pi, box_size, cutoff);
    // Calculate the hash
    const uint32_t hash = hashMorton(ci);
    // Create a hash combining the Morton hash and the batch index, so that atoms in the same cell
    // are contiguous
    const uint64_t hash_final = (static_cast<uint64_t>(hash) << 32) | i_batch;
    hash_keys[i_atom] = hash_final;
    hash_values[i_atom] = i_atom;
}

/*
 * @brief A buffer that is allocated and deallocated using the CUDA caching allocator from torch
 */
template <class T> struct CachedBuffer {
    explicit CachedBuffer(size_t size) : size_(size) {
        ptr_ = static_cast<T*>(at::cuda::CUDACachingAllocator::raw_alloc(size * sizeof(T)));
    }
    ~CachedBuffer() {
        at::cuda::CUDACachingAllocator::raw_delete(ptr_);
    }
    T* get() {
        return ptr_;
    }
    size_t size() {
        return size_;
    }

private:
    T* ptr_;
    size_t size_;
};

/*
 * @brief Sort the positions by hash, first by the cell assigned to each position and the batch
 * index
 * @param positions The positions of the atoms
 * @param batch The batch index of each atom
 * @param box_size The box vectors
 * @param cutoff The cutoff
 * @return A tuple of the sorted positions, sorted batch indices and the original indices of each
 * atom in the sorted list
 */
static auto sortPositionsByHash(const Tensor& positions, const Tensor& batch,
                                const Tensor& box_size, const Scalar& cutoff) {
    const int num_atoms = positions.size(0);
    auto hash_keys = CachedBuffer<uint64_t>(num_atoms);
    Tensor hash_values = empty({num_atoms}, positions.options().dtype(torch::kInt32));
    const int threads = 128;
    const int blocks = (num_atoms + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "assignHash", [&] {
        scalar_t cutoff_ = cutoff.to<scalar_t>();
        scalar3<scalar_t> box_size_ = {box_size[0][0].item<scalar_t>(),
                                       box_size[1][1].item<scalar_t>(),
                                       box_size[2][2].item<scalar_t>()};
        assignHash<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(positions), hash_keys.get(),
            get_accessor<int32_t, 1>(hash_values), get_accessor<int64_t, 1>(batch), box_size_,
            cutoff_, num_atoms);
    });
    // I have to use cub directly because thrust::sort_by_key is not compatible with graphs
    //  and torch::lexsort does not support uint64_t
    size_t tmp_storage_bytes = 0;
    auto d_keys_out = CachedBuffer<uint64_t>(num_atoms);
    auto d_values_out = CachedBuffer<int32_t>(num_atoms);
    auto* hash_values_ptr = hash_values.data_ptr<int32_t>();
    cub::DeviceRadixSort::SortPairs(nullptr, tmp_storage_bytes, hash_keys.get(), d_keys_out.get(),
                                    hash_values_ptr, d_values_out.get(), num_atoms, 0, 64, stream);
    auto tmp_storage = CachedBuffer<char>(tmp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(tmp_storage.get(), tmp_storage_bytes, hash_keys.get(),
                                    d_keys_out.get(), hash_values_ptr, d_values_out.get(),
                                    num_atoms, 0, 64, stream);
    cudaMemcpyAsync(hash_values_ptr, d_values_out.get(), num_atoms * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    Tensor sorted_positions = positions.index_select(0, hash_values);
    Tensor sorted_batch = batch.index_select(0, hash_values);
    return std::make_tuple(sorted_positions, sorted_batch, hash_values);
}

template <typename scalar_t>
__global__ void fillCellOffsetsD(const Accessor<scalar_t, 2> sorted_positions,
                                 const Accessor<int32_t, 1> sorted_indices,
                                 Accessor<int32_t, 1> cell_start, Accessor<int32_t, 1> cell_end,
                                 scalar3<scalar_t> box_size, scalar_t cutoff) {
    // Since positions are sorted by cell, for a given atom, if the previous atom is in a different
    // cell, then the current atom is the first atom in its cell We use this fact to fill the
    // cell_start and cell_end arrays
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= sorted_positions.size(0))
        return;
    const auto pi = fetchPosition(sorted_positions, i_atom);
    const int3 cell_dim = getCellDimensions(box_size, cutoff);
    const int icell = getCellIndex(getCell(pi, box_size, cutoff), cell_dim);
    int im1_cell;
    if (i_atom > 0) {
        int im1 = i_atom - 1;
        const auto pim1 = fetchPosition(sorted_positions, im1);
        im1_cell = getCellIndex(getCell(pim1, box_size, cutoff), cell_dim);
    } else {
        im1_cell = 0;
    }
    if (icell != im1_cell || i_atom == 0) {
        int n_cells = cell_start.size(0);
        cell_start[icell] = i_atom;
        if (i_atom > 0) {
            cell_end[im1_cell] = i_atom;
        }
    }
    if (i_atom == sorted_positions.size(0) - 1) {
        cell_end[icell] = i_atom + 1;
    }
}

/*
  @brief Fills the cell_start and cell_end arrays, identifying the first and last atom in each cell
  @param sorted_positions The positions sorted by cell
  @param sorted_indices The original indices of the sorted positions
  @param batch The batch index of each position
  @param box_size The box vectors
  @param cutoff The cutoff distance
  @return A tuple of cell_start and cell_end arrays
*/
static auto fillCellOffsets(const Tensor& sorted_positions, const Tensor& sorted_indices,
                            const Tensor& box_size, const Scalar& cutoff) {
    const TensorOptions options = sorted_positions.options();
    int3 cell_dim;
    AT_DISPATCH_FLOATING_TYPES(sorted_positions.scalar_type(), "fillCellOffsets", [&] {
        scalar_t cutoff_ = cutoff.to<scalar_t>();
        scalar3<scalar_t> box_size_ = {box_size[0][0].item<scalar_t>(),
                                       box_size[1][1].item<scalar_t>(),
                                       box_size[2][2].item<scalar_t>()};
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
        scalar3<scalar_t> box_size_ = {box_size[0][0].item<scalar_t>(),
                                       box_size[1][1].item<scalar_t>(),
                                       box_size[2][2].item<scalar_t>()};
        fillCellOffsetsD<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(sorted_positions), get_accessor<int32_t, 1>(sorted_indices),
            get_accessor<int32_t, 1>(cell_start), get_accessor<int32_t, 1>(cell_end), box_size_,
            cutoff_);
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

template <class scalar_t> struct Particle {
    int index;          // Index in the sorted arrays
    int original_index; // Index in the original arrays
    int batch;
    scalar3<scalar_t> position;
    scalar_t cutoff_upper2, cutoff_lower2;
};

struct CellList {
    Tensor cell_start, cell_end;
    Tensor original_indices;
    Tensor sorted_positions, sorted_batch;
};

CellList constructCellList(const Tensor& positions, const Tensor& batch, const Tensor& box_size,
                           const Scalar& cutoff) {
    // The algorithm for the cell list construction can be summarized in three separate steps:
    //         1. Hash (label) the particles according to the cell (bin) they lie in.
    //         2. Sort the particles and hashes using the hashes as the ordering label
    //         (technically this is known as sorting by key). So that particles with positions
    //         lying in the same cell become contiguous in memory.
    //         3. Identify where each cell starts and ends in the sorted particle positions
    //         array.
    const TensorOptions options = positions.options();
    CellList cl;
    // Steps 1 and 2
    std::tie(cl.sorted_positions, cl.sorted_batch, cl.original_indices) =
        sortPositionsByHash(positions, batch, box_size, cutoff);
    // Step 3
    std::tie(cl.cell_start, cl.cell_end) =
        fillCellOffsets(cl.sorted_positions, cl.original_indices, box_size, cutoff);
    return cl;
}

template <class scalar_t> struct CellListAccessor {
    Accessor<int32_t, 1> cell_start, cell_end;
    Accessor<int32_t, 1> original_indices;
    Accessor<scalar_t, 2> sorted_positions;
    Accessor<int64_t, 1> sorted_batch;

    explicit CellListAccessor(const CellList& cl)
        : cell_start(get_accessor<int32_t, 1>(cl.cell_start)),
          cell_end(get_accessor<int32_t, 1>(cl.cell_end)),
          original_indices(get_accessor<int32_t, 1>(cl.original_indices)),
          sorted_positions(get_accessor<scalar_t, 2>(cl.sorted_positions)),
          sorted_batch(get_accessor<int64_t, 1>(cl.sorted_batch)) {
    }
};

/*
 * @brief Add a pair of particles to the pair list. If necessary, also add the transpose pair.
 * @param list The pair list
 * @param i The index of the first particle
 * @param j The index of the second particle
 * @param distance2 The squared distance between the particles
 * @param delta The vector between the particles
 */
template <class scalar_t>
__device__ void addNeighborPair(PairListAccessor<scalar_t>& list, const int i, const int j,
                                scalar_t distance2, scalar3<scalar_t> delta) {
    const bool requires_transpose = list.include_transpose and (j != i);
    const int ni = thrust::max(i, j);
    const int nj = thrust::min(i, j);
    const scalar_t delta_sign = (ni == i) ? scalar_t(1.0) : scalar_t(-1.0);
    const scalar_t distance = sqrt_(distance2);
    delta = {delta_sign * delta.x, delta_sign * delta.y, delta_sign * delta.z};
    addAtomPairToList(list, ni, nj, delta, distance, requires_transpose);
}

/*
 * @brief Add to the pair list all neighbors of particle i_atom in cell j_cell
 * @param i_atom The Information of the particle for which we are adding neighbors
 * @param j_cell The index of the cell in which we are looking for neighbors
 * @param cl The cell list
 * @param box_size The box size
 * @param list The pair list
 */
template <class scalar_t>
__device__ void addNeighborsForCell(const Particle<scalar_t>& i_atom, int j_cell,
                                    const CellListAccessor<scalar_t>& cl,
                                    scalar3<scalar_t> box_size, PairListAccessor<scalar_t>& list) {
    const auto first_particle = cl.cell_start[j_cell];
    if (first_particle != -1) { // Continue only if there are particles in this cell
        const auto last_particle = cl.cell_end[j_cell];
        for (int cur_j = first_particle; cur_j < last_particle; cur_j++) {
            const auto j_batch = cl.sorted_batch[cur_j];
            // Particles are sorted by batch after cell, so we can break early here
            if (j_batch > i_atom.batch) {
                break;
            }
            if ((j_batch == i_atom.batch) and
                ((cur_j < i_atom.index) or (list.loop and cur_j == i_atom.index))) {
                const auto position_j = fetchPosition(cl.sorted_positions, cur_j);
                const auto delta = rect::compute_distance<scalar_t>(i_atom.position, position_j,
                                                                    list.use_periodic, box_size);
                const scalar_t distance2 =
                    delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                if ((distance2 < i_atom.cutoff_upper2 and distance2 >= i_atom.cutoff_lower2) or
                    (list.loop and cur_j == i_atom.index)) {
                    const int orj = cl.original_indices[cur_j];
                    addNeighborPair(list, i_atom.original_index, orj, distance2, delta);
                } // endif
            }     // endif
        }         // endfor
    }             // endif
}

// Traverse the cell list for each atom and find the neighbors
template <typename scalar_t>
__global__ void traverseCellList(const CellListAccessor<scalar_t> cell_list,
                                 PairListAccessor<scalar_t> list, int num_atoms,
                                 scalar3<scalar_t> box_size, scalar_t cutoff_lower,
                                 scalar_t cutoff_upper) {
    // Each atom traverses the cells around it and finds the neighbors
    // Atoms for all batches are placed in the same cell list, but other batches are ignored while
    // traversing
    Particle<scalar_t> i_atom;
    i_atom.index = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom.index >= num_atoms) {
        return;
    }
    i_atom.original_index = cell_list.original_indices[i_atom.index];
    i_atom.batch = cell_list.sorted_batch[i_atom.index];
    i_atom.position = fetchPosition(cell_list.sorted_positions, i_atom.index);
    i_atom.cutoff_lower2 = cutoff_lower * cutoff_lower;
    i_atom.cutoff_upper2 = cutoff_upper * cutoff_upper;
    const int3 cell_i = getCell(i_atom.position, box_size, cutoff_upper);
    const int3 cell_dim = getCellDimensions(box_size, cutoff_upper);
    // Loop over the 27 cells around the current cell
    for (int i = 0; i < 27; i++) {
        const int neighbor_cell = getNeighborCellIndex(cell_i, i, cell_dim);
        addNeighborsForCell(i_atom, neighbor_cell, cell_list, box_size, list);
    }
}

class AutogradCellCUDA : public Function<AutogradCellCUDA> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& positions, const Tensor& batch,
                               const Tensor& box_size, bool use_periodic,
                               const Scalar& cutoff_lower, const Scalar& cutoff_upper,
                               const Scalar& max_num_pairs, bool loop, bool include_transpose) {
        // This module computes the pair list for a given set of particles, which may be in multiple
        // batches. The strategy is to first compute a cell list for all particles, and then
        // traverse the cell list for each particle to construct a pair list.
        checkInput(positions, batch);
        TORCH_CHECK(box_size.dim() == 2, "Expected \"box_size\" to have two dimensions");
        TORCH_CHECK(box_size.size(0) == 3 && box_size.size(1) == 3,
                    "Expected \"box_size\" to have shape (3, 3)");
        TORCH_CHECK(box_size[0][1].item<double>() == 0 && box_size[0][2].item<double>() == 0 &&
                        box_size[1][0].item<double>() == 0 && box_size[1][2].item<double>() == 0 &&
                        box_size[2][0].item<double>() == 0 && box_size[2][1].item<double>() == 0,
                    "Expected \"box_size\" to be diagonal");
        const auto max_num_pairs_ = max_num_pairs.toInt();
        TORCH_CHECK(max_num_pairs_ > 0, "Expected \"max_num_neighbors\" to be positive");
        const int num_atoms = positions.size(0);
        const auto cell_list = constructCellList(positions, batch, box_size, cutoff_upper);
        PairList list(max_num_pairs_, positions.options(), loop, include_transpose, use_periodic);
        const auto stream = getCurrentCUDAStream(positions.get_device());
        { // Traverse the cell list to find the neighbors
            const CUDAStreamGuard guard(stream);
            AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "forward", [&] {
                const scalar_t cutoff_upper_ = cutoff_upper.to<scalar_t>();
                TORCH_CHECK(cutoff_upper_ > 0, "Expected cutoff_upper to be positive");
                const scalar_t cutoff_lower_ = cutoff_lower.to<scalar_t>();
                const scalar3<scalar_t> box_size_ = {box_size[0][0].item<scalar_t>(),
                                                     box_size[1][1].item<scalar_t>(),
                                                     box_size[2][2].item<scalar_t>()};
                PairListAccessor<scalar_t> list_accessor(list);
                CellListAccessor<scalar_t> cell_list_accessor(cell_list);
                const int threads = 64;
                const int blocks = (num_atoms + threads - 1) / threads;
                traverseCellList<<<blocks, threads, 0, stream>>>(cell_list_accessor, list_accessor,
                                                                 num_atoms, box_size_,
                                                                 cutoff_lower_, cutoff_upper_);
            });
        }
        ctx->save_for_backward({list.neighbors, list.deltas, list.distances});
        ctx->saved_data["num_atoms"] = num_atoms;
        return {list.neighbors, list.deltas, list.distances, list.i_curr_pair};
    }

    static tensor_list backward(AutogradContext* ctx, const tensor_list& grad_inputs) {
        return common_backward(ctx, grad_inputs);
    }
};
#endif
