import numpy as np
from Utils import int_to_cell


# ----------------------------
# Reachability Controller Synthesis
# ----------------------------

def compute_optimal_reachability(safe_mask, target_mask, T, dfa_matrix, label_map, n_dfa, n_u):
    n_total = len(safe_mask)
    V = np.full(n_total, np.inf)  # V = [inf, inf, ..., inf]

    # Initialize Target
    initial_wins = np.where(target_mask & safe_mask)[0]
    V[initial_wins] = 0
    solved_mask = (V != np.inf)
    # solved_mask = safe_mask 
    print(f"DEBUG: Initial Targets: {len(initial_wins)}")
    
    iteration = 0
    while True:
        iteration += 1
        candidates = np.where(safe_mask & (~solved_mask))[0]
        newly_solved = []

        # DEBUG: Track failure reasons for the first candidate
        # Set to True for no debug info
        debug_done = True

        for ps in candidates:
            cell_idx = ps // n_dfa
            q_idx = ps % n_dfa

            # Check all actions
            for u_idx in range(n_u):
                succs = T.get((cell_idx, u_idx))
                if succs is None: continue

                all_succs_solved = True

                # Debug variables
                fail_reason = ""

                for s_cell in succs:
                    r_idx = label_map[s_cell]
                    q_next = dfa_matrix[q_idx, r_idx]

                    if q_next == -1:
                        all_succs_solved = False
                        fail_reason = "Hit Trap"
                        break

                    s_ps = s_cell * n_dfa + q_next

                    if not solved_mask[s_ps]:
                        all_succs_solved = False
                        # Check if it was a self-loop failure or external failure
                        if s_ps == ps:
                            fail_reason = "Self-Loop to Unsolved"
                        else:
                            fail_reason = f"Hit Unsolved Neighbor {s_ps}"
                        break

                if all_succs_solved:
                    newly_solved.append(ps)
                    break
                elif not debug_done and iteration > 1:
                    # Print why the FIRST action of the FIRST candidate failed in iter 2
                    # This tells us why expansion stopped
                    print(f"DEBUG: State {ps} (Cell {int_to_cell(cell_idx)}) Action {u_idx} Failed: {fail_reason}")

            if len(newly_solved) > 0 and not debug_done and iteration > 1:
                 # If we found a solution for this state, we don't need to debug it
                 pass
            elif not debug_done and iteration > 1:
                 # We finished checking a state and found NO valid actions. Stop debugging this iter.
                 debug_done = True

        if not newly_solved:
            break

        V[newly_solved] = iteration
        solved_mask[newly_solved] = True

        print(f"   Opt Reach Iter {iteration}: Added {len(newly_solved)} states.")

    return V, solved_mask