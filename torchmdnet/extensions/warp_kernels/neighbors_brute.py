"""Warp kernel for brute-force neighbor list computation.

Uses triangular indexing: 1 thread per unique pair (i, j) where i > j.
Supports triclinic PBC, batching, self-loops, and transpose pairs.
"""

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_neighbors_brute(dtype: str):
    """Generate a brute-force neighbor list kernel for the given float dtype."""
    dtype_wp = get_wp_fp_dtype(dtype)

    def neighbors_brute_fwd(
        positions: wp.array(ndim=2, dtype=dtype_wp),
        batch: wp.array(ndim=1, dtype=wp.int64),
        box: wp.array(ndim=1, dtype=dtype_wp),
        neighbors: wp.array(ndim=2, dtype=wp.int64),
        deltas: wp.array(ndim=2, dtype=dtype_wp),
        distances: wp.array(ndim=1, dtype=dtype_wp),
        counter: wp.array(ndim=1, dtype=wp.int32),
        n_atoms: int,
        num_all_pairs: int,
        max_pairs: int,
        box_batch_stride: int,
        use_periodic: int,
        include_transpose: int,
        loop_flag: int,
        cutoff_lower2: float,
        cutoff_upper2: float,
    ):
        idx = wp.tid()
        if idx >= num_all_pairs:
            return

        half = positions.dtype(0.5)
        one = positions.dtype(1.0)
        eight = positions.dtype(8.0)

        f_idx = positions.dtype(idx)
        sqrt_val = wp.sqrt(one + eight * f_idx)

        i = 0
        j = 0

        if loop_flag != 0:
            # With self-loops: j <= i, num_pairs = n*(n+1)/2
            row_f = wp.floor((positions.dtype(-1.0) + sqrt_val) * half)
            i = int(row_f)
            # Correction for floating-point edge cases (overflow-safe triangular number)
            if i % 2 == 0:
                t_i = (i / 2) * (i + 1)
            else:
                t_i = i * ((i + 1) / 2)
            if t_i > idx:
                i = i - 1
            # Recompute T(i) after correction
            if i % 2 == 0:
                t_i = (i / 2) * (i + 1)
            else:
                t_i = i * ((i + 1) / 2)
            j = idx - t_i
        else:
            # Without self-loops: j < i, num_pairs = n*(n-1)/2
            row_f = wp.floor((one + sqrt_val) * half)
            i = int(row_f)
            if i % 2 == 0:
                t_i = (i / 2) * (i - 1)
            else:
                t_i = i * ((i - 1) / 2)
            if t_i > idx:
                i = i - 1
            if i % 2 == 0:
                t_i = (i / 2) * (i - 1)
            else:
                t_i = i * ((i - 1) / 2)
            j = idx - t_i

        if i < 0 or i >= n_atoms or j < 0:
            return
        if loop_flag != 0:
            if j > i:
                return
        else:
            if j >= i:
                return

        batch_i = batch[i]
        batch_j = batch[j]
        if batch_i != batch_j:
            return

        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        dz = positions[i, 2] - positions[j, 2]

        if use_periodic != 0:
            box_base = int(batch_i) * box_batch_stride
            b00 = box[box_base]
            b10 = box[box_base + 3]
            b11 = box[box_base + 4]
            b20 = box[box_base + 6]
            b21 = box[box_base + 7]
            b22 = box[box_base + 8]

            # 3-step triclinic minimum image convention
            scale3 = wp.round(dz / b22)
            dx = dx - scale3 * b20
            dy = dy - scale3 * b21
            dz = dz - scale3 * b22

            scale2 = wp.round(dy / b11)
            dx = dx - scale2 * b10
            dy = dy - scale2 * b11

            scale1 = wp.round(dx / b00)
            dx = dx - scale1 * b00

        dist2 = dx * dx + dy * dy + dz * dz
        is_self = i == j

        if dist2 >= cutoff_upper2:
            return
        if (not is_self) and dist2 < cutoff_lower2:
            return

        dist = wp.sqrt(dist2)

        if include_transpose != 0 and (not is_self):
            write_idx = wp.atomic_add(counter, 0, 2)
            if write_idx + 1 < max_pairs:
                neighbors[0, write_idx] = wp.int64(i)
                neighbors[1, write_idx] = wp.int64(j)
                deltas[write_idx, 0] = dx
                deltas[write_idx, 1] = dy
                deltas[write_idx, 2] = dz
                distances[write_idx] = dist

                neighbors[0, write_idx + 1] = wp.int64(j)
                neighbors[1, write_idx + 1] = wp.int64(i)
                deltas[write_idx + 1, 0] = -dx
                deltas[write_idx + 1, 1] = -dy
                deltas[write_idx + 1, 2] = -dz
                distances[write_idx + 1] = dist
        else:
            write_idx = wp.atomic_add(counter, 0, 1)
            if write_idx < max_pairs:
                neighbors[0, write_idx] = wp.int64(i)
                neighbors[1, write_idx] = wp.int64(j)
                deltas[write_idx, 0] = dx
                deltas[write_idx, 1] = dy
                deltas[write_idx, 2] = dz
                distances[write_idx] = dist

    return wp.Kernel(
        neighbors_brute_fwd,
        key=f"neighbors_brute_fwd_{dtype}",
        module=wp.get_module(f"neighbors_brute_fwd_{dtype}"),
    )


neighbors_brute_fwd_fp32 = generate_neighbors_brute("float32")
neighbors_brute_fwd_fp64 = generate_neighbors_brute("float64")

add_module("neighbors_brute_fwd", ["float32"], neighbors_brute_fwd_fp32)
add_module("neighbors_brute_fwd", ["float64"], neighbors_brute_fwd_fp64)
