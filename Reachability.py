from cuda_ops import run_reachability_gpu


# ----------------------------
# Reachability Controller Synthesis (CUDA)
# ----------------------------

def compute_optimal_reachability(safe_mask, target_mask, T_offsets, T_counts, T_successors, dfa_matrix, label_map, n_dfa, n_u):
    return run_reachability_gpu(
        safe_mask, target_mask,
        T_offsets, T_counts, T_successors,
        dfa_matrix, label_map,
        n_dfa, n_u
    )