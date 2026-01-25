from cuda_ops import run_safety_game_gpu


# ----------------------------
# Implicit Safety Controller (CUDA)
# ----------------------------

def compute_safety_implicit_parallel(T_offsets, T_counts, T_successors, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u):
    """
    Compute the maximal safe set (greatest fixed point) using CUDA kernels.
    """
    return run_safety_game_gpu(
        T_offsets, T_counts, T_successors,
        dfa_matrix, label_map,
        bad_q_indices, n_cells, n_dfa, n_u
    )