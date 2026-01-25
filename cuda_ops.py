import math
import numpy as np
from numba import cuda


def flatten_transition_table(T, n_states, n_inputs):
    """
    Convert transition dict T[(state, action)] -> successors into CSR-like flat arrays.

    Returns:
        T_offsets: int32 array, shape (n_states * n_inputs + 1,)
        T_counts: int32 array, shape (n_states * n_inputs,)
        T_successors: int32 array, shape (sum of all successor counts,)
    """
    n_pairs = n_states * n_inputs
    counts = np.zeros(n_pairs, dtype=np.int64)

    for (state_idx, u_idx), succs in T.items():
        flat_idx = state_idx * n_inputs + u_idx
        counts[flat_idx] = len(succs)

    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Transition table too large for int32 offsets.")

    T_offsets = offsets.astype(np.int32)
    T_counts = counts.astype(np.int32)
    T_successors = np.empty(T_offsets[-1], dtype=np.int32)

    write_ptr = T_offsets[:-1].copy()
    for (state_idx, u_idx), succs in T.items():
        flat_idx = state_idx * n_inputs + u_idx
        start = write_ptr[flat_idx]
        end = start + len(succs)
        T_successors[start:end] = np.asarray(succs, dtype=np.int32)
        write_ptr[flat_idx] = end

    return T_offsets, T_counts, T_successors


def _flatten_dfa_matrix(dfa_matrix, n_dfa=None):
    dfa_matrix = np.asarray(dfa_matrix, dtype=np.int32)
    if dfa_matrix.ndim == 1:
        if n_dfa is None:
            raise ValueError("n_dfa must be provided when dfa_matrix is flat.")
        if dfa_matrix.size % n_dfa != 0:
            raise ValueError("Flat dfa_matrix size is not divisible by n_dfa.")
        n_inputs = dfa_matrix.size // n_dfa
        return dfa_matrix, n_dfa, n_inputs
    if dfa_matrix.ndim != 2:
        raise ValueError("dfa_matrix must be 2D or 1D before flattening.")
    n_dfa, n_inputs = dfa_matrix.shape
    return dfa_matrix.reshape(-1), n_dfa, n_inputs


def _ensure_label_map(label_map):
    label_map = np.asarray(label_map, dtype=np.int32).reshape(-1)
    return label_map


@cuda.jit
def _count_transitions_kernel(
    x1_bins, x2_bins, x3_bins,
    U_flat, w_center, radius_w,
    tau, x1_min, x2_min, x3_min,
    d_x1, d_x2, d_x3,
    Nx1, Nx2, Nx3, n_u,
    counts
):
    cell_idx = cuda.grid(1)
    n_states = Nx1 * Nx2 * Nx3
    if cell_idx >= n_states:
        return

    stride_x1 = Nx2 * Nx3
    stride_x2 = Nx3

    i = cell_idx // stride_x1
    rem = cell_idx - i * stride_x1
    j = rem // stride_x2
    k = rem - j * stride_x2

    c_x1 = 0.5 * (x1_bins[i] + x1_bins[i + 1])
    c_x2 = 0.5 * (x2_bins[j] + x2_bins[j + 1])
    c_x3 = 0.5 * (x3_bins[k] + x3_bins[k + 1])

    radius_x0 = 0.5 * d_x1
    radius_x1 = 0.5 * d_x2
    radius_x2 = 0.5 * d_x3

    for u_idx in range(n_u):
        base = u_idx * 2
        v = U_flat[base]
        omega = U_flat[base + 1]

        xn_0 = c_x1 + tau * (v * math.cos(c_x3) + w_center[0])
        xn_1 = c_x2 + tau * (v * math.sin(c_x3) + w_center[1])
        xn_2 = c_x3 + tau * (omega + w_center[2])

        term1_0 = radius_x0 + (tau * abs(v)) * radius_x2
        term1_1 = radius_x1 + (tau * abs(v)) * radius_x2
        term1_2 = radius_x2

        term2_0 = tau * radius_w[0]
        term2_1 = tau * radius_w[1]
        term2_2 = tau * radius_w[2]

        d_x_succ0 = term1_0 + term2_0
        d_x_succ1 = term1_1 + term2_1
        d_x_succ2 = term1_2 + term2_2

        lower_0 = xn_0 - d_x_succ0
        lower_1 = xn_1 - d_x_succ1
        lower_2 = xn_2 - d_x_succ2

        upper_0 = xn_0 + d_x_succ0
        upper_1 = xn_1 + d_x_succ1
        upper_2 = xn_2 + d_x_succ2

        if lower_2 < -math.pi and upper_2 >= -math.pi:
            lower_2 += 2.0 * math.pi
        elif lower_2 < -math.pi and upper_2 < -math.pi:
            lower_2 += 2.0 * math.pi
            upper_2 += 2.0 * math.pi
        elif upper_2 > math.pi and lower_2 <= math.pi:
            upper_2 -= 2.0 * math.pi
        elif upper_2 > math.pi and lower_2 > math.pi:
            upper_2 -= 2.0 * math.pi
            lower_2 -= 2.0 * math.pi

        i_min = int(math.floor((lower_0 - x1_min) / d_x1))
        i_max = int(math.floor((upper_0 - x1_min) / d_x1))

        j_min = int(math.floor((lower_1 - x2_min) / d_x2))
        j_max = int(math.floor((upper_1 - x2_min) / d_x2))

        k_min = int(math.floor((lower_2 - x3_min) / d_x3))
        k_max = int(math.floor((upper_2 - x3_min) / d_x3))

        if i_max < 0 or i_min >= Nx1 or j_max < 0 or j_min >= Nx2:
            counts[cell_idx * n_u + u_idx] = 0
            continue

        if i_min < 0:
            i_min = 0
        if i_max > Nx1 - 1:
            i_max = Nx1 - 1

        if j_min < 0:
            j_min = 0
        if j_max > Nx2 - 1:
            j_max = Nx2 - 1

        k_count = 0
        if k_min <= k_max:
            k_start = 0 if k_min < 0 else k_min
            k_end = (Nx3 - 1) if k_max > (Nx3 - 1) else k_max
            if k_start <= k_end:
                k_count = k_end - k_start + 1
        else:
            s1_start = 0 if k_min < 0 else k_min
            s1_end = Nx3 - 1
            if s1_start <= s1_end:
                k_count += s1_end - s1_start + 1

            s2_start = 0
            s2_end = (Nx3 - 1) if k_max > (Nx3 - 1) else k_max
            if s2_start <= s2_end:
                k_count += s2_end - s2_start + 1

        if k_count <= 0:
            counts[cell_idx * n_u + u_idx] = 0
            continue

        count = (i_max - i_min + 1) * (j_max - j_min + 1) * k_count
        counts[cell_idx * n_u + u_idx] = count


@cuda.jit
def _fill_transitions_kernel(
    x1_bins, x2_bins, x3_bins,
    U_flat, w_center, radius_w,
    tau, x1_min, x2_min, x3_min,
    d_x1, d_x2, d_x3,
    Nx1, Nx2, Nx3, n_u,
    offsets, successors
):
    cell_idx = cuda.grid(1)
    n_states = Nx1 * Nx2 * Nx3
    if cell_idx >= n_states:
        return

    stride_x1 = Nx2 * Nx3
    stride_x2 = Nx3

    i = cell_idx // stride_x1
    rem = cell_idx - i * stride_x1
    j = rem // stride_x2
    k = rem - j * stride_x2

    c_x1 = 0.5 * (x1_bins[i] + x1_bins[i + 1])
    c_x2 = 0.5 * (x2_bins[j] + x2_bins[j + 1])
    c_x3 = 0.5 * (x3_bins[k] + x3_bins[k + 1])

    radius_x0 = 0.5 * d_x1
    radius_x1 = 0.5 * d_x2
    radius_x2 = 0.5 * d_x3

    for u_idx in range(n_u):
        base = u_idx * 2
        v = U_flat[base]
        omega = U_flat[base + 1]

        xn_0 = c_x1 + tau * (v * math.cos(c_x3) + w_center[0])
        xn_1 = c_x2 + tau * (v * math.sin(c_x3) + w_center[1])
        xn_2 = c_x3 + tau * (omega + w_center[2])

        term1_0 = radius_x0 + (tau * abs(v)) * radius_x2
        term1_1 = radius_x1 + (tau * abs(v)) * radius_x2
        term1_2 = radius_x2

        term2_0 = tau * radius_w[0]
        term2_1 = tau * radius_w[1]
        term2_2 = tau * radius_w[2]

        d_x_succ0 = term1_0 + term2_0
        d_x_succ1 = term1_1 + term2_1
        d_x_succ2 = term1_2 + term2_2

        lower_0 = xn_0 - d_x_succ0
        lower_1 = xn_1 - d_x_succ1
        lower_2 = xn_2 - d_x_succ2

        upper_0 = xn_0 + d_x_succ0
        upper_1 = xn_1 + d_x_succ1
        upper_2 = xn_2 + d_x_succ2

        if lower_2 < -math.pi and upper_2 >= -math.pi:
            lower_2 += 2.0 * math.pi
        elif lower_2 < -math.pi and upper_2 < -math.pi:
            lower_2 += 2.0 * math.pi
            upper_2 += 2.0 * math.pi
        elif upper_2 > math.pi and lower_2 <= math.pi:
            upper_2 -= 2.0 * math.pi
        elif upper_2 > math.pi and lower_2 > math.pi:
            upper_2 -= 2.0 * math.pi
            lower_2 -= 2.0 * math.pi

        i_min = int(math.floor((lower_0 - x1_min) / d_x1))
        i_max = int(math.floor((upper_0 - x1_min) / d_x1))

        j_min = int(math.floor((lower_1 - x2_min) / d_x2))
        j_max = int(math.floor((upper_1 - x2_min) / d_x2))

        k_min = int(math.floor((lower_2 - x3_min) / d_x3))
        k_max = int(math.floor((upper_2 - x3_min) / d_x3))

        if i_max < 0 or i_min >= Nx1 or j_max < 0 or j_min >= Nx2:
            continue

        if i_min < 0:
            i_min = 0
        if i_max > Nx1 - 1:
            i_max = Nx1 - 1

        if j_min < 0:
            j_min = 0
        if j_max > Nx2 - 1:
            j_max = Nx2 - 1

        p = offsets[cell_idx * n_u + u_idx]

        if k_min <= k_max:
            k_start = 0 if k_min < 0 else k_min
            k_end = (Nx3 - 1) if k_max > (Nx3 - 1) else k_max
            if k_start > k_end:
                continue

            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    for kk in range(k_start, k_end + 1):
                        successors[p] = ii * stride_x1 + jj * stride_x2 + kk
                        p += 1
        else:
            s1_start = 0 if k_min < 0 else k_min
            s1_end = Nx3 - 1
            s2_start = 0
            s2_end = (Nx3 - 1) if k_max > (Nx3 - 1) else k_max

            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    if s1_start <= s1_end:
                        for kk in range(s1_start, s1_end + 1):
                            successors[p] = ii * stride_x1 + jj * stride_x2 + kk
                            p += 1
                    if s2_start <= s2_end:
                        for kk in range(s2_start, s2_end + 1):
                            successors[p] = ii * stride_x1 + jj * stride_x2 + kk
                            p += 1


@cuda.jit
def _safety_kernel(
    T_offsets, T_counts, T_successors,
    dfa_flat, label_map,
    n_dfa, n_dfa_inputs, n_u,
    safe_in, safe_out, changed
):
    ps = cuda.grid(1)
    if ps >= safe_in.size:
        return

    if not safe_in[ps]:
        safe_out[ps] = False
        return

    cell_idx = ps // n_dfa
    q_idx = ps - cell_idx * n_dfa

    action_found = False
    for u_idx in range(n_u):
        trans_idx = cell_idx * n_u + u_idx
        count = T_counts[trans_idx]
        if count == 0:
            continue

        offset = T_offsets[trans_idx]
        ok = True
        for t in range(count):
            s_cell = T_successors[offset + t]
            r_idx = label_map[s_cell]
            q_next = dfa_flat[q_idx * n_dfa_inputs + r_idx]
            if q_next == -1:
                ok = False
                break

            s_ps = s_cell * n_dfa + q_next
            if not safe_in[s_ps]:
                ok = False
                break

        if ok:
            action_found = True
            break

    safe_out[ps] = action_found
    if action_found != safe_in[ps]:
        cuda.atomic.max(changed, 0, 1)


@cuda.jit
def _reachability_kernel(
    T_offsets, T_counts, T_successors,
    dfa_flat, label_map,
    n_dfa, n_dfa_inputs, n_u,
    safe_mask, target_mask,
    V_in, V_out, changed
):
    ps = cuda.grid(1)
    if ps >= V_in.size:
        return

    if not safe_mask[ps]:
        V_out[ps] = V_in[ps]
        return

    if target_mask[ps]:
        V_out[ps] = 0.0
        return

    cell_idx = ps // n_dfa
    q_idx = ps - cell_idx * n_dfa

    best = math.inf
    for u_idx in range(n_u):
        trans_idx = cell_idx * n_u + u_idx
        count = T_counts[trans_idx]
        if count == 0:
            continue

        offset = T_offsets[trans_idx]
        max_val = -math.inf
        valid = True
        for t in range(count):
            s_cell = T_successors[offset + t]
            r_idx = label_map[s_cell]
            q_next = dfa_flat[q_idx * n_dfa_inputs + r_idx]
            if q_next == -1:
                valid = False
                break

            s_ps = s_cell * n_dfa + q_next
            if not safe_mask[s_ps]:
                valid = False
                break

            v_val = V_in[s_ps]
            if v_val > max_val:
                max_val = v_val

        if valid and max_val < best:
            best = max_val

    new_val = V_in[ps]
    if best < math.inf:
        candidate = 1.0 + best
        if candidate < new_val:
            new_val = candidate

    V_out[ps] = new_val
    if new_val != V_in[ps]:
        cuda.atomic.max(changed, 0, 1)


def compute_transitions_cuda(x1_bins, x2_bins, x3_bins, U_vals, w_bounds, tau, Nx1, Nx2, Nx3):
    x1_bins = np.asarray(x1_bins, dtype=np.float64)
    x2_bins = np.asarray(x2_bins, dtype=np.float64)
    x3_bins = np.asarray(x3_bins, dtype=np.float64)

    U_flat = np.asarray(U_vals, dtype=np.float64).reshape(-1)
    w_bounds = np.asarray(w_bounds, dtype=np.float64)

    w_center = np.mean(w_bounds, axis=1).astype(np.float64)
    radius_w = (0.5 * (w_bounds[:, 1] - w_bounds[:, 0])).astype(np.float64)

    d_x1 = float(x1_bins[1] - x1_bins[0])
    d_x2 = float(x2_bins[1] - x2_bins[0])
    d_x3 = float(x3_bins[1] - x3_bins[0])

    x1_min = float(x1_bins[0])
    x2_min = float(x2_bins[0])
    x3_min = float(x3_bins[0])

    n_states = Nx1 * Nx2 * Nx3
    n_u = len(U_flat) // 2
    n_pairs = n_states * n_u

    counts_device = cuda.device_array(n_pairs, dtype=np.int32)

    threads = 128
    blocks = (n_states + threads - 1) // threads

    _count_transitions_kernel[blocks, threads](
        x1_bins, x2_bins, x3_bins,
        U_flat, w_center, radius_w,
        tau, x1_min, x2_min, x3_min,
        d_x1, d_x2, d_x3,
        Nx1, Nx2, Nx3, n_u,
        counts_device
    )

    counts = counts_device.copy_to_host().astype(np.int64)
    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Transition buffer too large for int32 indexing.")

    T_offsets = offsets.astype(np.int32)
    T_counts = counts.astype(np.int32)
    T_successors = np.empty(T_offsets[-1], dtype=np.int32)

    offsets_device = cuda.to_device(T_offsets)
    successors_device = cuda.device_array((T_offsets[-1],), dtype=np.int32)

    _fill_transitions_kernel[blocks, threads](
        x1_bins, x2_bins, x3_bins,
        U_flat, w_center, radius_w,
        tau, x1_min, x2_min, x3_min,
        d_x1, d_x2, d_x3,
        Nx1, Nx2, Nx3, n_u,
        offsets_device, successors_device
    )

    T_successors[:] = successors_device.copy_to_host()

    return T_offsets, T_counts, T_successors


def compute_safety_implicit_cuda(T_offsets, T_counts, T_successors, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u):
    dfa_flat, n_dfa_states, n_dfa_inputs = _flatten_dfa_matrix(dfa_matrix, n_dfa=n_dfa)
    if n_dfa_states != n_dfa:
        raise ValueError("n_dfa does not match dfa_matrix shape.")

    label_map = _ensure_label_map(label_map)

    total_states = n_cells * n_dfa
    safe_mask = np.ones(total_states, dtype=np.bool_)
    for q_bad in bad_q_indices:
        safe_mask[q_bad::n_dfa] = False

    print(f"Initial Candidate Safe States: {np.sum(safe_mask)}")

    T_offsets = np.asarray(T_offsets, dtype=np.int32)
    T_counts = np.asarray(T_counts, dtype=np.int32)
    T_successors = np.asarray(T_successors, dtype=np.int32)

    d_T_offsets = cuda.to_device(T_offsets)
    d_T_counts = cuda.to_device(T_counts)
    d_T_successors = cuda.to_device(T_successors)
    d_dfa_flat = cuda.to_device(dfa_flat)
    d_label_map = cuda.to_device(label_map)

    safe_in = cuda.to_device(safe_mask)
    safe_out = cuda.device_array_like(safe_in)
    changed = cuda.device_array(1, dtype=np.int32)

    threads = 128
    blocks = (total_states + threads - 1) // threads

    iteration = 0
    while True:
        iteration += 1
        changed.copy_to_device(np.zeros(1, dtype=np.int32))

        _safety_kernel[blocks, threads](
            d_T_offsets, d_T_counts, d_T_successors,
            d_dfa_flat, d_label_map,
            n_dfa, n_dfa_inputs, n_u,
            safe_in, safe_out, changed
        )

        if changed.copy_to_host()[0] == 0:
            safe_in = safe_out
            print(f"   Safety Iter {iteration}: Removed 0, Remaining {np.sum(safe_in.copy_to_host())}")
            break

        safe_in, safe_out = safe_out, safe_in
        print(f"   Safety Iter {iteration}: Remaining {np.sum(safe_in.copy_to_host())}")

    return safe_in.copy_to_host()


def compute_optimal_reachability_cuda(safe_mask, target_mask, T_offsets, T_counts, T_successors, dfa_matrix, label_map, n_dfa, n_u, max_iters=10000):
    dfa_flat, n_dfa_states, n_dfa_inputs = _flatten_dfa_matrix(dfa_matrix, n_dfa=n_dfa)
    if n_dfa_states != n_dfa:
        raise ValueError("n_dfa does not match dfa_matrix shape.")

    label_map = _ensure_label_map(label_map)

    total_states = len(safe_mask)
    V = np.full(total_states, np.inf, dtype=np.float32)

    initial_wins = np.where(target_mask & safe_mask)[0]
    V[initial_wins] = 0.0
    print(f"DEBUG: Initial Targets: {len(initial_wins)}")

    T_offsets = np.asarray(T_offsets, dtype=np.int32)
    T_counts = np.asarray(T_counts, dtype=np.int32)
    T_successors = np.asarray(T_successors, dtype=np.int32)

    d_T_offsets = cuda.to_device(T_offsets)
    d_T_counts = cuda.to_device(T_counts)
    d_T_successors = cuda.to_device(T_successors)
    d_dfa_flat = cuda.to_device(dfa_flat)
    d_label_map = cuda.to_device(label_map)

    d_safe_mask = cuda.to_device(np.asarray(safe_mask, dtype=np.bool_))
    d_target_mask = cuda.to_device(np.asarray(target_mask, dtype=np.bool_))

    V_in = cuda.to_device(V)
    V_out = cuda.device_array_like(V_in)
    changed = cuda.device_array(1, dtype=np.int32)

    threads = 128
    blocks = (total_states + threads - 1) // threads

    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iters:
            print("Reached max iterations in reachability.")
            break

        changed.copy_to_device(np.zeros(1, dtype=np.int32))

        _reachability_kernel[blocks, threads](
            d_T_offsets, d_T_counts, d_T_successors,
            d_dfa_flat, d_label_map,
            n_dfa, n_dfa_inputs, n_u,
            d_safe_mask, d_target_mask,
            V_in, V_out, changed
        )

        if changed.copy_to_host()[0] == 0:
            V_in = V_out
            break

        V_in, V_out = V_out, V_in

    V_host = V_in.copy_to_host()
    win_mask = np.isfinite(V_host)

    return V_host, win_mask


def run_abstraction_kernel(x1_bins, x2_bins, x3_bins, U_vals, w_bounds, tau, Nx1, Nx2, Nx3):
    return compute_transitions_cuda(x1_bins, x2_bins, x3_bins, U_vals, w_bounds, tau, Nx1, Nx2, Nx3)


def run_safety_game_gpu(T_offsets, T_counts, T_successors, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u):
    return compute_safety_implicit_cuda(
        T_offsets, T_counts, T_successors,
        dfa_matrix, label_map,
        bad_q_indices, n_cells, n_dfa, n_u
    )


def run_reachability_gpu(safe_mask, target_mask, T_offsets, T_counts, T_successors, dfa_matrix, label_map, n_dfa, n_u):
    return compute_optimal_reachability_cuda(
        safe_mask, target_mask,
        T_offsets, T_counts, T_successors,
        dfa_matrix, label_map,
        n_dfa, n_u
    )
