import multiprocessing
from joblib import Parallel, delayed
import numpy as np


# ----------------------------
# Optimized Safety Controller Synthesis (process-based)
# ----------------------------

def _check_safety_chunk(chunk_indices, R_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    """Evaluate safety for a subset of product states and return indices to remove."""
    to_remove = []
    for ps in chunk_indices:
        if not R_mask[ps]:
            continue

        cell_idx = ps // n_dfa
        q_idx = ps % n_dfa

        action_found = False
        for u_idx in range(n_u):
            succs = T.get((cell_idx, u_idx))
            if succs is None:
                continue

            # Assume action is safe until a bad successor is found
            for s_cell in succs:
                r_idx = label_map[s_cell]
                q_next = dfa_matrix[q_idx, r_idx]
                if q_next == -1:
                    action_found = False
                    break

                s_ps = s_cell * n_dfa + q_next
                if not R_mask[s_ps]:
                    action_found = False
                    break
            else:
                action_found = True

            if action_found:
                break

        if not action_found:
            to_remove.append(ps)

    return to_remove


# ----------------------------
# Implicit Safety Controller (process parallel)
# ----------------------------

def compute_safety_implicit_parallel(T, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u):
    """
    Compute the maximal safe set (greatest fixed point) using process-based parallelism.

    Each iteration filters unsafe product states by checking whether some action keeps
    all successors inside the current candidate set. The loop stops at the fixed point.
    """

    num_cores = multiprocessing.cpu_count()

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

        chunk_size = max(1, len(candidates) // num_cores)
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

        results = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(_check_safety_chunk)(chunk, R_mask, T, dfa_matrix, label_map, n_dfa, n_u)
            for chunk in chunks
        )

        removed = [ps for res in results for ps in res]
        if not removed:
            print(f"   Safety Iter {iteration}: Removed 0, Remaining {np.sum(R_mask)}")
            break

        R_mask[removed] = False
        print(f"   Safety Iter {iteration}: Removed {len(removed)}, Remaining {np.sum(R_mask)}")

    return R_mask