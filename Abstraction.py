from Parameters import U_vals, w_bounds, tau, Nx1, Nx2, Nx3, x1_bins, x2_bins, x3_bins
from cuda_ops import run_abstraction_kernel

def build_transition_table_parallel():
    """
    Builds the full transition table using CUDA kernels.

    Returns:
        T_offsets, T_counts, T_successors (CSR-like flattened transition structure)
    """
    print(f"Building T on GPU... (Total States: {Nx1 * Nx2 * Nx3})")
    T_offsets, T_counts, T_successors = run_abstraction_kernel(
        x1_bins, x2_bins, x3_bins, U_vals, w_bounds, tau, Nx1, Nx2, Nx3
    )
    return T_offsets, T_counts, T_successors