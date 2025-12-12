from Utils import cell_to_int, int_to_cell
from Parameters import x1_range, x2_range, x3_range, U_vals, w_bounds, tau, Nx1, Nx2, Nx3, x1_bins, x2_bins, x3_bins, TOTAL_CELLS
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import math


def compute_transitions_chunk(cell_indices, U_vals, x1_bins, x2_bins, x3_bins, w_bounds, tau, Nx1, Nx2, Nx3):
    """
    Compute symbolic transitions for a chunk of cells using the Growth Bound method.

    Inputs:
    - cell_indices: iterable of integer cell ids (flat indexing) assigned to this worker.
    - U_vals: list of control inputs (v, omega) pairs.
    - x1_bins, x2_bins, x3_bins: discretization bin edges for each dimension.
    - w_bounds: array of disturbance bounds [[w1_min,w1_max],...].
    - tau: sampling time.
    - Nx1, Nx2, Nx3: grid dimensions.

    Output:
    - local_T: dict mapping (cell_idx, u_idx) -> tuple(successor_cell_indices)
    """
    local_T = {}

    # 1. Pre-compute constants
    
    # Grid steps (widths)
    d_x1 = x1_bins[1] - x1_bins[0]
    d_x2 = x2_bins[1] - x2_bins[0]
    d_x3 = x3_bins[1] - x3_bins[0]
    dx_vec = np.array([d_x1, d_x2, d_x3])

    # Half-widths (radius) for growth bound calculation
    radius_x = 0.5 * dx_vec

    # Disturbance radius
    w_width = w_bounds[:, 1] - w_bounds[:, 0]
    radius_w = 0.5 * w_width
    w_center = np.mean(w_bounds, axis=1) # w_c in MATLAB


    # 2. Iterate over assigned cells
    for cell_idx in cell_indices:
        i, j, k = int_to_cell(cell_idx)

        # Center of the current cell x_c
        c_x1 = 0.5 * (x1_bins[i] + x1_bins[i+1])
        c_x2 = 0.5 * (x2_bins[j] + x2_bins[j+1])
        c_x3 = 0.5 * (x3_bins[k] + x3_bins[k+1])
        x_c = np.array([c_x1, c_x2, c_x3])

        for u_idx, u in enumerate(U_vals):
            v, omega = u

            # --- A. Nominal Dynamics (x_c_succ) ---
            # f(x_c, u, w_c)
            # x1_next
            xn_0 = x_c[0] + tau * (v * math.cos(x_c[2]) + w_center[0])
            # x2_next
            xn_1 = x_c[1] + tau * (v * math.sin(x_c[2]) + w_center[1])
            # x3_next
            xn_2 = x_c[2] + tau * (omega + w_center[2])

            x_c_succ = np.array([xn_0, xn_1, xn_2])

            # --- B. Growth Bound (d_x_succ) ---
            # Jf_x = [[1, 0, tau*|v|], [0, 1, tau*|v|], [0, 0, 1]]
            # Jf_w = diag(tau, tau, tau)

            # 0.5 * Jf_x * d_x
            term1_0 = 1.0 * radius_x[0] + 0.0 + (tau * abs(v)) * radius_x[2]
            term1_1 = 0.0 + 1.0 * radius_x[1] + (tau * abs(v)) * radius_x[2]
            term1_2 = 0.0 + 0.0 + 1.0 * radius_x[2]

            # 0.5 * Jf_w * d_w
            term2_0 = tau * radius_w[0]
            term2_1 = tau * radius_w[1]
            term2_2 = tau * radius_w[2]

            d_x_succ = np.array([
                term1_0 + term2_0,
                term1_1 + term2_1,
                term1_2 + term2_2
            ])

            # --- C. Reachable Set Interval ---
            lower = x_c_succ - d_x_succ
            upper = x_c_succ + d_x_succ

            # --- D. Angle Wrapping Logic ---
            if lower[2] < -math.pi and upper[2] >= -math.pi:
                lower[2] += 2 * math.pi
            elif lower[2] < -math.pi and upper[2] < -math.pi:
                lower[2] += 2 * math.pi
                upper[2] += 2 * math.pi
            elif upper[2] > math.pi and lower[2] <= math.pi:
                upper[2] -= 2 * math.pi
            elif upper[2] > math.pi and lower[2] > math.pi:
                upper[2] -= 2 * math.pi
                lower[2] -= 2 * math.pi

            # --- E. Map to Grid Indices ---
            # X1
            i_min = int(np.floor((lower[0] - x1_range[0]) / d_x1))
            i_max = int(np.floor((upper[0] - x1_range[0]) / d_x1))

            # X2
            j_min = int(np.floor((lower[1] - x2_range[0]) / d_x2))
            j_max = int(np.floor((upper[1] - x2_range[0]) / d_x2))

            # X3
            k_min = int(np.floor((lower[2] - x3_range[0]) / d_x3))
            k_max = int(np.floor((upper[2] - x3_range[0]) / d_x3))

            # Check Out of Bounds (Spatial)
            # If the reachable set is completely outside the map, the transition is invalid.
            # If it partially overlaps, we clip it
            if i_max < 0 or i_min >= Nx1 or j_max < 0 or j_min >= Nx2:
                continue

            i_min = max(0, i_min)
            i_max = min(Nx1 - 1, i_max)
            j_min = max(0, j_min)
            j_max = min(Nx2 - 1, j_max)

            # Handle Theta indices with wrapping
            # If k_min > k_max, it means the interval wrapped around the edge
            k_indices = []

            # Normalize k indices to 0..Nx3-1 just in case slight float errors pushed them out
            if k_min <= k_max:
                # Standard Contiguous Range
                # Clip to valid 0..Nx3-1
                k_start = max(0, k_min)
                k_end = min(Nx3 - 1, k_max)
                k_indices = list(range(k_start, k_end + 1))
            else:
                # Wrapped Range: [k_min...End] U [Start...k_max]
                # Clip segments independently

                # Segment 1: k_min to End
                s1_start = max(0, k_min)
                s1_end = Nx3 - 1
                if s1_start <= s1_end:
                    k_indices.extend(range(s1_start, s1_end + 1))

                # Segment 2: Start to k_max
                s2_start = 0
                s2_end = min(Nx3 - 1, k_max)
                if s2_start <= s2_end:
                    k_indices.extend(range(s2_start, s2_end + 1))

            if not k_indices:
                continue

            # --- F. Build Successor List ---
            succs = []
            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    for kk in k_indices:
                        succs.append(cell_to_int(ii, jj, kk))

            if succs:
                local_T[(cell_idx, u_idx)] = tuple(succs)

    return local_T

def build_transition_table_parallel():
    """
    Computes and builds the full transition table (T) in parallel using
    multiprocessing.
    """
    num_cores = multiprocessing.cpu_count()
    all_indices = range(TOTAL_CELLS)
    chunk_size = len(all_indices) // num_cores + 1
    chunks = [all_indices[i:i + chunk_size] for i in range(0, len(all_indices), chunk_size)]

    print(f"Building T on {num_cores} cores... (Total States: {TOTAL_CELLS})")

    results = Parallel(n_jobs=num_cores)(
        delayed(compute_transitions_chunk)(
            chunk, U_vals, x1_bins, x2_bins, x3_bins, w_bounds, tau, Nx1, Nx2, Nx3
        ) for chunk in chunks
    )

    T = {}
    for res in results:
        T.update(res)
    return T