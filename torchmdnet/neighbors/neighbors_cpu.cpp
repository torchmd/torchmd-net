#include <torch/extension.h>
#include <tuple>

namespace torch {

static const double radius = 10;

static double get_distance(const Tensor v1, const Tensor v2) {

    const Tensor diffs = v2 - v1;
    const double distance = diffs.square().sum().sqrt().item<double>();

    return distance;
}

static std::tuple<Tensor, Tensor, Tensor> forward(const Tensor& positions) {

    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    const int num_atoms = positions.size(0);
    const int num_neighbors = num_atoms;

    const Tensor indices = arange(0, num_neighbors, positions.options().dtype(kInt32));

    const Tensor vectors = index_select(positions, 0, indices);

    Tensor max_coords = zeros({3}, kFloat64);

    for(unsigned iter = 0; iter < num_atoms; iter++) {
        const Tensor vector = vectors[iter];

        max_coords[0] = max(vector[0], max_coords[0]);
        max_coords[1] = max(vector[1], max_coords[1]);
        max_coords[2] = max(vector[2], max_coords[2]);
    }

    const int num_partitions_x = ceil((max_coords[0]).item<double>() / radius) + 2;
    const int num_partitions_y = ceil((max_coords[1]).item<double>() / radius) + 2;
    const int num_partitions_z = ceil((max_coords[2]).item<double>() / radius) + 2;
    const int num_partitions = num_partitions_x * num_partitions_y * num_partitions_z;

    auto coords_to_offset = [num_partitions_x, num_partitions_y, num_partitions_z](const unsigned coord_x, const unsigned coord_y, const unsigned coord_z) {
        const unsigned index = (num_partitions_x - 1) * ((num_partitions_y - 1) * coord_z + coord_y) + coord_x;
        return index;
    };

    std::vector<std::vector<int>> hashes;
    hashes.resize(num_partitions);
    std::vector<int> neighbor_rows;
    std::vector<int> neighbor_cols;
    std::vector<double> neighbor_distances;

    for(int iter = 0; iter < num_atoms; iter++) {
        const Tensor vector = vectors[iter];

        const unsigned coord_x = round((vector[0]).item<double>() / radius) + 1;
        const unsigned coord_y = round((vector[1]).item<double>() / radius) + 1;
        const unsigned coord_z = round((vector[2]).item<double>() / radius) + 1;

        for(unsigned iter_x = coord_x - 1; iter_x <= coord_x + 1; iter_x++) {
            for(unsigned iter_y = coord_y - 1; iter_y <= coord_y + 1; iter_y++) {
                for(unsigned iter_z = coord_z - 1; iter_z <= coord_z + 1; iter_z++) {
                    const unsigned index = coords_to_offset(iter_x, iter_y, iter_z);

                    const std::vector<int> hash = hashes[index];
                    for(int neighbor: hash) {
                        const double distance = get_distance(vectors[iter], vectors[neighbor]);
                        if(distance < radius) {
                            neighbor_rows.push_back(iter);
                            neighbor_cols.push_back(neighbor);
                            neighbor_distances.push_back(distance);
                        }
                    }
                }
            }
        }
        const unsigned index = coords_to_offset(coord_x, coord_y, coord_z);
        hashes[index].push_back(iter);
    }

    const Tensor neighbor_rows_tensor = from_blob(neighbor_rows.data(), {neighbor_rows.size()}, TensorOptions().dtype(kInt32)).clone();
    const Tensor neighbor_cols_tensor = from_blob(neighbor_cols.data(), {neighbor_cols.size()}, TensorOptions().dtype(kInt32)).clone();
    const Tensor neighbor_distances_tensor = from_blob(neighbor_distances.data(), {neighbor_distances.size()}, TensorOptions().dtype(kFloat64)).clone();

    return {neighbor_rows_tensor, neighbor_cols_tensor, neighbor_distances_tensor};
}

TORCH_LIBRARY_IMPL(neighbors, CPU, m) {
    m.impl("get_neighbor_list", &forward);
}

}
