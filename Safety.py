from joblib import Parallel, delayed
import multiprocessing
import numpy as np

# ----------------------------
# Optimized Safety Controller Synthesis
# ----------------------------

def check_safety_chunk(chunk_indices, R_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    to_remove = []
    for ps in chunk_indices:
        if not R_mask[ps]: continue

        cell_idx = ps // n_dfa
        q_idx = ps % n_dfa

        can_save = False

        for u_idx in range(n_u):
            succs = T.get((cell_idx, u_idx))
            if succs is None: continue

            action_safe = True
            for s_cell in succs:
                r_idx = label_map[s_cell]
                q_next = dfa_matrix[q_idx, r_idx]

                if q_next == -1:
                    action_safe = False
                    break

                s_ps = s_cell * n_dfa + q_next
                if not R_mask[s_ps]:
                    action_safe = False
                    break

            if action_safe:
                can_save = True
                break

        if not can_save:
            to_remove.append(ps)

    return to_remove


# ----------------------------
# Implicit Safety Controller
# ----------------------------

def compute_safety_implicit_parallel(T, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u):
    """
    Compute the maximal safe set (greatest fixed point) using an implicit iteration.

    The procedure starts with a candidate mask `R_mask` marking all product states as safe (except explicit trap states).
    In each iteration we inspect all currently marked states and remove those for which no control action exists that keeps *all* successors in the candidate set. The loop stops when no more states are removed (fixed point).

    Inputs:
    - T: transition dictionary mapping (cell_idx, u_idx) -> tuple(successor_cell_indices)
    - dfa_matrix: integer matrix mapping (q_idx, region_idx) -> next q_idx
    - label_map: array mapping cell_idx -> region_idx
    - bad_q_indices: list of DFA indices that correspond to trap states
    - n_cells, n_dfa, n_u: dimensions

    Output:
    - R_mask: boolean mask of length n_cells * n_dfa marking winning (safe) product states.
    """
    # Ensure NUM_CORES is available
    num_cores = multiprocessing.cpu_count()

    # Initialize: All valid states except explicit traps
    R_mask = np.ones(n_cells * n_dfa, dtype=bool)
    for q_bad in bad_q_indices:
        R_mask[q_bad::n_dfa] = False

    print(f"Initial Candidate Safe States: {np.sum(R_mask)}")

    iteration = 0
    while True:
        iteration += 1
        candidates = np.where(R_mask)[0]

        if len(candidates) == 0:
            break

        chunk_size = len(candidates) // num_cores + 1
        chunks = [candidates[i:i+chunk_size] for i in range(0, len(candidates), chunk_size)]

        results = Parallel(n_jobs=num_cores, backend="threading")(
            delayed(check_safety_chunk)(
                chunk, R_mask, T, dfa_matrix, label_map, n_dfa, n_u
            ) for chunk in chunks
        )

        removed_count = 0
        for res in results:
            if res:
                R_mask[res] = False
                removed_count += len(res)

        print(f"   Safety Iter {iteration}: Removed {removed_count}, Remaining {np.sum(R_mask)}")
        if removed_count == 0:
            break

    return R_mask