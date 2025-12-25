import multiprocessing
from joblib import Parallel, delayed
import numpy as np


# ----------------------------
# Reachability Controller Synthesis (process-based)
# ----------------------------

def _reachability_chunk(chunk_indices, solved_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    """Return states in the chunk that can reach the target in one more step."""
    newly_solved = []
    for ps in chunk_indices:
        cell_idx = ps // n_dfa
        q_idx = ps % n_dfa

        for u_idx in range(n_u):
            succs = T.get((cell_idx, u_idx))
            if succs is None:
                continue

            for s_cell in succs:
                r_idx = label_map[s_cell]
                q_next = dfa_matrix[q_idx, r_idx]
                if q_next == -1:
                    break

                s_ps = s_cell * n_dfa + q_next
                if not solved_mask[s_ps]:
                    break
            else:
                newly_solved.append(ps)
                break

    return newly_solved


def compute_optimal_reachability(safe_mask, target_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    n_total = len(safe_mask)
    V = np.full(n_total, np.inf)

    initial_wins = np.where(target_mask & safe_mask)[0]
    V[initial_wins] = 0
    solved_mask = (V != np.inf)
    print(f"DEBUG: Initial Targets: {len(initial_wins)}")

    num_cores = multiprocessing.cpu_count()
    iteration = 0
    while True:
        iteration += 1
        candidates = np.where(safe_mask & (~solved_mask))[0]
        if len(candidates) == 0:
            break

        chunk_size = max(1, len(candidates) // num_cores)
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

        results = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(_reachability_chunk)(chunk, solved_mask, T, dfa_matrix, label_map, n_dfa, n_u)
            for chunk in chunks
        )

        newly_solved = [ps for res in results for ps in res]
        if not newly_solved:
            break

        V[newly_solved] = iteration
        solved_mask[newly_solved] = True

        print(f"   Opt Reach Iter {iteration}: Added {len(newly_solved)} states.")

    return V, solved_mask