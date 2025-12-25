import multiprocessing
from joblib import Parallel, delayed
import numpy as np

# ----------------------------
# Controller Synthesis (unique controllers, process-based)
# ----------------------------

def _controller_chunk(chunk_indices, V, winning_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    """Compute optimal actions for a subset of winning product states."""
    chunk_controller = {}
    for ps in chunk_indices:
        if V[ps] == 0:
            continue

        cell_idx = ps // n_dfa
        q_idx = ps % n_dfa

        best_u = -1
        best_worst_case_val = np.inf

        for u_idx in range(n_u):
            succs = T.get((cell_idx, u_idx))
            if succs is None:
                continue

            current_action_max_val = -np.inf
            is_valid_action = True

            for s_cell in succs:
                r_idx = label_map[s_cell]
                q_next = dfa_matrix[q_idx, r_idx]
                if q_next == -1:
                    is_valid_action = False
                    break

                s_ps = s_cell * n_dfa + q_next
                if not winning_mask[s_ps]:
                    is_valid_action = False
                    break

                v_val = V[s_ps]
                if v_val > current_action_max_val:
                    current_action_max_val = v_val

            if is_valid_action and current_action_max_val < best_worst_case_val:
                best_worst_case_val = current_action_max_val
                best_u = u_idx

        if best_u != -1:
            chunk_controller[ps] = best_u

    return chunk_controller


def synthesize_optimal_controller(V, winning_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    """
    Synthesizes a controller that minimizes the worst-case steps to the target.
    Policy: u(s) = argmin_u ( max_{s' in Post(s,u)} V(s') )
    """
    print("Synthesizing Optimal Controller...")

    winning_indices = np.where(winning_mask)[0]
    if len(winning_indices) == 0:
        return {}

    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(winning_indices) // num_cores)
    chunks = [winning_indices[i:i + chunk_size] for i in range(0, len(winning_indices), chunk_size)]

    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(_controller_chunk)(chunk, V, winning_mask, T, dfa_matrix, label_map, n_dfa, n_u)
        for chunk in chunks
    )

    controller = {}
    for partial in results:
        controller.update(partial)

    return controller