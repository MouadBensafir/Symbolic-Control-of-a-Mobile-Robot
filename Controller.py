import numpy as np

# ----------------------------
# Controller Synthesis (unique controllers)
# ----------------------------

def synthesize_optimal_controller(V, winning_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    """
    Synthesizes a controller that minimizes the worst-case steps to the target.
    Policy: u(s) = argmin_u ( max_{s' in Post(s,u)} V(s') )
    """
    print("Synthesizing Optimal Controller...")
    controller = {}

    winning_indices = np.where(winning_mask)[0]

    for ps in winning_indices:
        # If already at target (V=0), no action needed (or stay put)
        if V[ps] == 0:
            continue

        cell_idx = ps // n_dfa
        q_idx = ps % n_dfa

        best_u = -1
        best_worst_case_val = np.inf

        for u_idx in range(n_u):
            succs = T.get((cell_idx, u_idx))
            if succs is None: continue

            # Evaluate Robust Cost of this action
            # The cost of an action is the MAX Value of all possible outcomes (Worst Case)
            current_action_max_val = -1
            is_valid_action = True

            for s_cell in succs:
                r_idx = label_map[s_cell]
                q_next = dfa_matrix[q_idx, r_idx]

                if q_next == -1:
                    is_valid_action = False; break

                s_ps = s_cell * n_dfa + q_next

                # If successor is unsafe/unreachable, this action is invalid
                # if not winning_mask[s_ps]:
                #     is_valid_action = False; break

                # Track the worst (highest) value among successors
                if V[s_ps] > current_action_max_val:
                    current_action_max_val = V[s_ps]

            # Optimization: Pick action with lowest worst-case cost
            if is_valid_action:
                if current_action_max_val < best_worst_case_val:
                    best_worst_case_val = current_action_max_val
                    best_u = u_idx

        if best_u != -1:
            controller[ps] = best_u

    return controller