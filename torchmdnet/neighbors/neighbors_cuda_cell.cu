#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
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
    // This version works with batches
    // Batch contains the molecule index for each atom in positions
    // Neighbors are only calculated within the same molecule
    // Batch is a 1D tensor of size (N_atoms)
    // Batch is assumed to be sorted and starts at zero.
    // Batch is assumed to be contiguous
    // Batch is assumed to be of type torch::kInt32
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
    TORCH_CHECK(batch.dtype() == torch::kInt32, "Expected \"batch\" to be of type torch::kInt32");
}

// Encodes an integer lower than 1024 as a 32 bit integer by filling every third bit.
inline __host__ __device__ uint encodeMorton(const uint& i) {
    uint x = i;
    x &= 0x3ff;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
}

// Fuse three 10 bit numbers in 32 bits, producing a Z order Morton hash
inline __host__ __device__ uint hashMorton(int3 ci) {
    return encodeMorton(ci.x) | (encodeMorton(ci.y) << 1) | (encodeMorton(ci.z) << 2);
}

// Use Minimum Image Convention to take a point to the unit cell
__device__ auto takeToUnitCell(float3 p, float3 box_size) {
    p.x = p.x - floorf(p.x / box_size.x + float(0.5)) * box_size.x;
    p.y = p.y - floorf(p.y / box_size.y + float(0.5)) * box_size.y;
    p.z = p.z - floorf(p.z / box_size.z + float(0.5)) * box_size.z;
    return p;
}

// Get the number of cells in each dimension
__host__ __device__ int3 getNumberCells(float3 box_size, float cutoff) {
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

// Get the cell coordinates of a point
__device__ int3 getCell(float3 p, float3 box_size, float cutoff) {
    p = takeToUnitCell(p, box_size);
    int cx = floorf(p.x / cutoff);
    int cy = floorf(p.y / cutoff);
    int cz = floorf(p.z / cutoff);
    int3 cell_dim = getNumberCells(box_size, cutoff);
    if (cx == cell_dim.x)
        cx = 0;
    if (cy == cell_dim.y)
        cy = 0;
    if (cz == cell_dim.z)
        cz = 0;
    return make_int3(cx, cy, cz);
}

// Assign a hash to each atom based on its position and batch.
// This hash is such that atoms in the same cell and batch have the same hash.
template <typename scalar_t>
__global__ void assignHash(const Accessor<scalar_t, 2> positions, uint64_t* hash_keys,
                           Accessor<int32_t, 1> hash_values, const Accessor<int32_t, 1> batch,
                           float3 box_size, float cutoff, int32_t num_atoms) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    const int32_t i_batch = batch[i_atom];
    // Move to the unit cell
    float3 pi = make_float3(positions[i_atom][0], positions[i_atom][1], positions[i_atom][2]);
    auto ci = getCell(pi, box_size, cutoff);
    // Calculate the hash
    const int32_t hash = hashMorton(ci);
    // Create a hash combining the Morton hash and the batch index, so that atoms in the same batch
    // are contiguous
    const int64_t hash_final = (static_cast<int64_t>(i_batch) << 32) | hash;
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
        return static_cast<T*>(at::cuda::getCUDADeviceAllocator()->raw_allocate(num_elements * sizeof(T)));
    }
    void deallocate(T* ptr, size_t) {
      at::cuda::getCUDADeviceAllocator()->raw_deallocate(ptr);
    }
};

// Sort the positions by hash, based on the cell assigned to each position and the batch index
static auto sortPositionsByHash(const Tensor& positions, const Tensor& batch, float3 box_size,
                                float cutoff) {
    const int num_atoms = positions.size(0);
    const auto options = positions.options();
    thrust::device_vector<uint64_t> hash_keys(num_atoms);
    Tensor hash_values = empty({num_atoms}, options.dtype(torch::kInt32));
    const int threads = 128;
    const int blocks = (num_atoms + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "assignHash", [&] {
        assignHash<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(positions), thrust::raw_pointer_cast(hash_keys.data()),
            get_accessor<int32_t, 1>(hash_values), get_accessor<int32_t, 1>(batch), box_size,
            cutoff, num_atoms);
    });

    std::cout << "hash_values: " << hash_values << std::endl;
    std::cout << "hash_keys: " << std::endl;
    for (int i = 0; i < num_atoms; i++) {
      uint64_t hi = hash_keys[i];
      std::cout << std::bitset<64>(hi) << std::endl;
    }

    // Sort positions by hash_values using thrust
    thrust::device_ptr<int32_t> index_ptr(hash_values.data_ptr<int32_t>());
    // pytorch allocator
    CudaAllocator<char> allocator;
    // Adapted for thrust
    thrust::sort_by_key(thrust::cuda::par.on(stream), hash_keys.begin(), hash_keys.end(),
                        index_ptr);
    std::cout << "sorted hash_values: " << hash_values << std::endl;
    // Print values of hash_keys in binary
    std::cout << "sorted hash_keys: " << std::endl;
    for (int i = 0; i < num_atoms; i++) {
      uint64_t hi = hash_keys[i];
      std::cout << std::bitset<64>(hi) << std::endl;
    }

    Tensor sorted_positions = positions.index_select(0, hash_values);
    return std::make_tuple(sorted_positions, hash_values);
}

__device__ int getCellIndex(int3 cell, int3 cell_dim) {
    return cell.x + cell_dim.x * (cell.y + cell_dim.y * cell.z);
}

template <typename scalar_t>
__global__ void fillCellOffsetsD(const Accessor<scalar_t, 2> sorted_positions,
				 const Accessor<int32_t, 1> sorted_indices,
                                 Accessor<int32_t, 2> cell_start, Accessor<int32_t, 2> cell_end,
                                 const Accessor<int32_t, 1> batch, float3 box_size, float cutoff) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= sorted_positions.size(0))
        return;
    const int32_t i_batch = batch[sorted_indices[i_atom]];
    const float3 pi = make_float3(sorted_positions[i_atom][0], sorted_positions[i_atom][1],
                                  sorted_positions[i_atom][2]);
    const int3 cell_dim = getNumberCells(box_size, cutoff);
    const int icell = getCellIndex(getCell(pi, box_size, cutoff), cell_dim);
    int im1_cell;
    if (i_atom > 0) {
        int im1 = i_atom - 1;
        const float3 pim1 = make_float3(sorted_positions[im1][0], sorted_positions[im1][1],
                                        sorted_positions[im1][2]);
        im1_cell = getCellIndex(getCell(pim1, box_size, cutoff), cell_dim);
    } else {
        im1_cell = 0;
    }
    if (icell != im1_cell || i_atom == 0) {
      int n_cells = cell_start.size(0);
      if(icell>=n_cells or im1_cell>=n_cells) {
	printf("icell: %d, im1_cell: %d, n_cells: %d\n", icell, im1_cell, n_cells);
	return;
      }
        cell_start[icell][i_batch] = i_atom;
        if (i_atom > 0)
            cell_end[im1_cell][i_batch] = i_atom;
    }
    if(i_atom == sorted_positions.size(0) - 1) {
      cell_end[icell][i_batch] = i_atom + 1;
    }
}

// Fill the cell offsets for each batch, identifying the start and end of each cell for each batch
// in the sorted positions
static auto fillCellOffsets(const Tensor& sorted_positions,
			    const Tensor& sorted_indices,
			    const Tensor& batch, float3 box_size,
                            float cutoff) {
    const TensorOptions options = sorted_positions.options();
    const int num_batches = batch[-1].item<int32_t>() + 1;
    const int3 cell_dim = getNumberCells(box_size, cutoff);
    const int num_cells = cell_dim.x * cell_dim.y * cell_dim.z;
    const Tensor cell_start = full({num_cells, num_batches}, -1, options.dtype(torch::kInt));
    const Tensor cell_end = empty({num_cells, num_batches}, options.dtype(torch::kInt));
    std::cerr<<"num_cells: "<<num_cells<<std::endl;
    std::cerr<<"num_batches: "<<num_batches<<std::endl;
    const int threads = 128;
    const int blocks = (sorted_positions.size(0) + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(sorted_positions.scalar_type(), "fillCellOffsets", [&] {
        auto stream = at::cuda::getCurrentCUDAStream();
        fillCellOffsetsD<<<blocks, threads, 0, stream>>>(
            get_accessor<scalar_t, 2>(sorted_positions),
	    get_accessor<int32_t, 1>(sorted_indices),
	    get_accessor<int32_t, 2>(cell_start),
            get_accessor<int32_t, 2>(cell_end), get_accessor<int32_t, 1>(batch), box_size, cutoff);
    });
    std::cerr<<"cell_start: "<<cell_start<<std::endl;
    std::cerr<<"cell_end: "<<cell_end<<std::endl;
    return std::make_tuple(cell_start, cell_end);
}

// Fold a cell coordinate to the range [0, cell_dim)
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

// Traverse the cell list for each atom and find the neighbors
template <typename scalar_t>
__global__ void
forward_kernel(const Accessor<scalar_t, 2> sorted_positions,
               const Accessor<int32_t, 1> original_index, const Accessor<int32_t, 1> batch,
               const Accessor<int32_t, 2> cell_start, const Accessor<int32_t, 2> cell_end,
               Accessor<int32_t, 2> neighbors, Accessor<scalar_t, 2> deltas,
               Accessor<scalar_t, 1> distances, Accessor<int32_t, 1> i_curr_pair, int num_atoms,
               int num_pairs, float3 box_size, float cutoff) {
    const int32_t i_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_atom >= num_atoms)
        return;
    // Each batch has its own cell list, starting at cell_start[0][i_batch] and ending at
    // cell_start[ncells-1][i_batch] Each thread is responsible for a single atom Each thread will
    // loop over all atoms in the cell list of the current batch
    const int ori = original_index[i_atom];
    const int i_batch = batch[ori];
    float3 pi = make_float3(sorted_positions[i_atom][0], sorted_positions[i_atom][1],
                            sorted_positions[i_atom][2]);
    const int3 cell_i = getCell(pi, box_size, cutoff);
    const int3 cell_dim = getNumberCells(box_size, cutoff);
    const int i_cell_index = getCellIndex(cell_i, cell_dim);
    // Loop over the 27 cells around the current cell
    for (int i = 0; i < 27; i++) {
        auto cell_j = cell_i;
        cell_j.x += i % 3 - 1;
        cell_j.y += (i / 3) % 3 - 1;
        cell_j.z += i / 9 - 1;
        cell_j = getPeriodicCell(cell_j, cell_dim);
        int icellj = getCellIndex(cell_j, cell_dim);
        const int firstParticle = cell_start[icellj][i_batch];
        if (firstParticle != -1) { // Continue only if there are particles in this cell
            // Index of the last particle in the cell's list
            const int lastParticle = cell_end[icellj][i_batch];
            const int nincell = lastParticle - firstParticle;
            for (int j = 0; j < nincell; j++) {
                int cur_j = j + firstParticle;
                if (cur_j < i_atom) {
                    float3 pj = make_float3(sorted_positions[cur_j][0], sorted_positions[cur_j][1],
                                            sorted_positions[cur_j][2]);
                    const scalar_t dx = pi.x - pj.x;
                    const scalar_t dy = pi.y - pj.y;
                    const scalar_t dz = pi.z - pj.z;
                    const scalar_t distance2 = dx * dx + dy * dy + dz * dz;
                    if (distance2 < cutoff * cutoff) {
                        const int32_t i_pair = atomicAdd(&i_curr_pair[0], 1);
                        // We handle too many neighbors outside of the kernel
                        if (i_pair < neighbors.size(1)) {
                            neighbors[0][i_pair] = ori;
                            neighbors[1][i_pair] = original_index[cur_j];
                            deltas[i_pair][0] = dx;
                            deltas[i_pair][1] = dy;
                            deltas[i_pair][2] = dz;
                            distances[i_pair] = sqrt_(distance2);
                        }
                    }
                } // endfor
            }     // endif
        }         // endfor
    }
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx, const Tensor& positions, const Tensor& batch,
                               const Tensor& box_size, const Scalar& cutoff,
                               const Scalar& max_num_pairs, bool checkErrors) {

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
        float3 box_size_ = make_float3(box_size[0].item<float>(), box_size[1].item<float>(),
                                       box_size[2].item<float>());
        float cutoff_ = cutoff.toFloat();
        Tensor sorted_positions, hash_values;
        std::cerr << "before sortPositionsByHash" << std::endl;
        std::tie(sorted_positions, hash_values) =
            sortPositionsByHash(positions, batch, box_size_, cutoff_);
        cudaDeviceSynchronize();
        std::cerr << "after sortPositionsByHash" << std::endl;
        // Step 3
        Tensor cell_start, cell_end;
        std::cerr << "before fillCellOffsets" << std::endl;
        std::tie(cell_start, cell_end) =
	  fillCellOffsets(sorted_positions, hash_values,
			  batch, box_size_, cutoff_);
        cudaDeviceSynchronize();
        std::cerr << "after fillCellOffsets" << std::endl;
	cudaDeviceSynchronize();
	std::cerr << "Number of pairs: " << num_pairs << std::endl;
	cudaDeviceSynchronize();
	std::cerr <<"Allocating memory for neighbors" << std::endl;
        const Tensor neighbors = full({2, num_pairs}, -1, options.dtype(kInt32));
	std::cerr << "Allocating memory for deltas" << std::endl;
        const Tensor deltas = empty({num_pairs, 3}, options);
	std::cerr << "Allocating memory for distances" << std::endl;
        const Tensor distances = full(num_pairs, 0, options);
        const Tensor i_curr_pair = zeros(1, options.dtype(kInt32));
        cudaDeviceSynchronize();
        std::cerr << "before forward_kernel" << std::endl;
	const auto stream = getCurrentCUDAStream(positions.get_device());
        { // Use the cell list for each batch to find the neighbors
            const CUDAStreamGuard guard(stream);
            AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "forward", [&] {
                const int threads = 128;
                const int blocks = (num_atoms + threads - 1) / threads;
                forward_kernel<<<blocks, threads, 0, stream>>>(
                    get_accessor<scalar_t, 2>(sorted_positions),
                    get_accessor<int32_t, 1>(hash_values), get_accessor<int32_t, 1>(batch),
                    get_accessor<int32_t, 2>(cell_start), get_accessor<int32_t, 2>(cell_end),
                    get_accessor<int32_t, 2>(neighbors), get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 1>(distances), get_accessor<int32_t, 1>(i_curr_pair),
                    num_atoms, num_pairs, box_size_, cutoff_);
            });
        }
        cudaDeviceSynchronize();
        std::cerr << "after forward_kernel" << std::endl;
        // Synchronize and check the number of pairs found. Note that this is incompatible with CUDA
        // graphs
        if (checkErrors) {
	  std::cout<<"checking errors"<<std::endl;
            int num_found_pairs = i_curr_pair[0].item<int32_t>();
            TORCH_CHECK(num_found_pairs <= max_num_pairs_,
                        "Too many neighbor pairs found. Maximum is " +
                            std::to_string(max_num_pairs_),
                        " but found " + std::to_string(num_found_pairs));
	    std::cout<<"no errors"<<std::endl;
        }
	std::cout<<"before resize"<<std::endl;
        neighbors.resize_({2, i_curr_pair[0].item<int32_t>()});
        deltas.resize_({i_curr_pair[0].item<int32_t>(), 3});
        distances.resize_(i_curr_pair[0].item<int32_t>());
	std::cout<<"after resize"<<std::endl;
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
              const Scalar& cutoff, const Scalar& max_num_pairs, bool checkErrors) {
               const tensor_list results =
                   Autograd::apply(positions, batch, box_size, cutoff, max_num_pairs, checkErrors);
               return std::make_tuple(results[0], results[1], results[2]);
           });
}
