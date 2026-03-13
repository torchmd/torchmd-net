"""Warp kernel for cell-list based neighbor list computation.

Uses a 1-thread-per-atom approach: each thread determines its cell, iterates
over 27 neighbor cells, and checks all atoms within each neighbor cell.
Supports orthogonal PBC (diagonal box) only.
"""

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_neighbors_cell(dtype: str):
    """Generate a cell-list neighbor traversal kernel for the given float dtype."""
    dtype_wp = get_wp_fp_dtype(dtype)

    def neighbors_cell_fwd(
        sorted_positions: wp.array(ndim=2, dtype=dtype_wp),
        sorted_indices: wp.array(ndim=1, dtype=wp.int32),
        sorted_batch: wp.array(ndim=1, dtype=wp.int64),
        cell_start: wp.array(ndim=1, dtype=wp.int32),
        cell_end: wp.array(ndim=1, dtype=wp.int32),
        box_sizes: wp.array(ndim=1, dtype=dtype_wp),
        cell_dims: wp.array(ndim=1, dtype=wp.int32),
        neighbors: wp.array(ndim=2, dtype=wp.int64),
        deltas: wp.array(ndim=2, dtype=dtype_wp),
        distances: wp.array(ndim=1, dtype=dtype_wp),
        counter: wp.array(ndim=1, dtype=wp.int32),
        n_atoms: int,
        max_pairs: int,
        use_periodic: int,
        include_transpose: int,
        loop_flag: int,
        cutoff_lower2: float,
        cutoff_upper2: float,
    ):
        atom_idx = wp.tid()
        if atom_idx >= n_atoms:
            return

        my_orig = sorted_indices[atom_idx]
        my_x = sorted_positions[atom_idx, 0]
        my_y = sorted_positions[atom_idx, 1]
        my_z = sorted_positions[atom_idx, 2]
        my_batch = sorted_batch[atom_idx]

        bx = box_sizes[0]
        by = box_sizes[1]
        bz = box_sizes[2]

        nx = cell_dims[0]
        ny = cell_dims[1]
        nz = cell_dims[2]

        # Determine home cell from position
        f_nx = sorted_positions.dtype(nx)
        f_ny = sorted_positions.dtype(ny)
        f_nz = sorted_positions.dtype(nz)
        cell_sx = bx / f_nx
        cell_sy = by / f_ny
        cell_sz = bz / f_nz

        if use_periodic != 0:
            wx = my_x - wp.floor(my_x / bx) * bx
            wy = my_y - wp.floor(my_y / by) * by
            wz = my_z - wp.floor(my_z / bz) * bz
        else:
            half = sorted_positions.dtype(0.5)
            wx = my_x + half * bx
            wy = my_y + half * by
            wz = my_z + half * bz

        home_cx = wp.clamp(int(wx / cell_sx), 0, nx - 1)
        home_cy = wp.clamp(int(wy / cell_sy), 0, ny - 1)
        home_cz = wp.clamp(int(wz / cell_sz), 0, nz - 1)

        nyz = ny * nz

        for nc in range(27):
            di = nc % 3 - 1
            dj = (nc / 3) % 3 - 1
            dk = nc / 9 - 1

            ni = home_cx + di
            nj = home_cy + dj
            nk = home_cz + dk

            cell_valid = 1
            if use_periodic != 0:
                ni = (ni + nx) % nx
                nj = (nj + ny) % ny
                nk = (nk + nz) % nz
            else:
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    cell_valid = 0

            if cell_valid != 0:
                nc_flat = ni * nyz + nj * nz + nk
                ns = cell_start[nc_flat]
                ne = cell_end[nc_flat]

                for j_idx in range(ns, ne):
                    j_orig = sorted_indices[j_idx]
                    j_batch = sorted_batch[j_idx]

                    if j_batch != my_batch:
                        continue

                    # Index ordering to avoid double-counting
                    if include_transpose != 0:
                        if loop_flag == 0 and my_orig == j_orig:
                            continue
                    else:
                        if loop_flag != 0:
                            if my_orig < j_orig:
                                continue
                        else:
                            if my_orig <= j_orig:
                                continue

                    dx = my_x - sorted_positions[j_idx, 0]
                    dy = my_y - sorted_positions[j_idx, 1]
                    dz = my_z - sorted_positions[j_idx, 2]

                    if use_periodic != 0:
                        dx = dx - bx * wp.round(dx / bx)
                        dy = dy - by * wp.round(dy / by)
                        dz = dz - bz * wp.round(dz / bz)

                    dist2 = dx * dx + dy * dy + dz * dz
                    is_self = my_orig == j_orig

                    if dist2 >= cutoff_upper2:
                        continue
                    if (not is_self) and dist2 < cutoff_lower2:
                        continue

                    dist = wp.sqrt(dist2)

                    write_idx = wp.atomic_add(counter, 0, 1)
                    if write_idx < max_pairs:
                        neighbors[0, write_idx] = wp.int64(my_orig)
                        neighbors[1, write_idx] = wp.int64(j_orig)
                        deltas[write_idx, 0] = dx
                        deltas[write_idx, 1] = dy
                        deltas[write_idx, 2] = dz
                        distances[write_idx] = dist

    return wp.Kernel(
        neighbors_cell_fwd,
        key=f"neighbors_cell_fwd_{dtype}",
        module=wp.get_module(f"neighbors_cell_fwd_{dtype}"),
    )


neighbors_cell_fwd_fp32 = generate_neighbors_cell("float32")
neighbors_cell_fwd_fp64 = generate_neighbors_cell("float64")

add_module("neighbors_cell_fwd", ["float32"], neighbors_cell_fwd_fp32)
add_module("neighbors_cell_fwd", ["float64"], neighbors_cell_fwd_fp64)
