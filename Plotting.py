from Utils import cell_to_int, f_cont, q
from Parameters import x1_range, x2_range
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


# ----------------------------
# Trajectory Generation from initial State
# ----------------------------

def simulate(x0, dfa_init_idx, controller, dfa_matrix, label_map, n_dfa, U_vals, regions, state_map, a_accept, steps=100):
    """
    Simulates the system on a 2D grid. Stops if goal is reached.
    """
    traj = [x0]

    # Map Continuous x0 -> Discrete Cell -> Integer ID
    curr_cell = cell_to_int(*q(x0))
    curr_q = dfa_init_idx

    # Encode Product State correctly
    curr_ps = curr_cell * n_dfa + curr_q

    # Inverse map for debugging
    inv_state_map = {v: k for k, v in state_map.items()}

    print(f"Sim Start: Cell {curr_cell} (Q={curr_q}, {inv_state_map[curr_q]}) -> Product State {curr_ps}")

    for i in range(steps):
        # 0. Check if we are in an accepting state (Goal Reached)
        current_dfa_state_name = inv_state_map[curr_q]
        if current_dfa_state_name in a_accept:
             print(f"   Step {i}: Goal Reached! (DFA State: {current_dfa_state_name})")
             break

        # 1. Check if state has a valid control action
        if curr_ps not in controller:
            print(f"   Step {i}: No Valid Control Action! Stuck at Product State {curr_ps}")
            break

        # 2. Get Control Action
        u_idx = controller[curr_ps]
        u = U_vals[u_idx]

        # 3. Apply Continuous Dynamics
        x_curr = traj[-1]
        # Zero disturbance for nominal sim
        x_next = f_cont(x_curr, u, np.zeros(3))
        traj.append(x_next)

        # 4. Update Discrete State
        next_cell = cell_to_int(*q(x_next))

        # 5. Evolve DFA (Logic Step)
        if next_cell < len(label_map):
            r_idx = label_map[next_cell]
        else:
            print("   -> Went out of bounds!")
            break

        next_q = dfa_matrix[curr_q, r_idx]

        # 6. Check for Safety Violation
        if next_q == -1:
            print("   -> DFA Unsafe Trap State Reached!")
            break

        # 7. Update Loop State
        curr_q = next_q
        curr_cell = next_cell

        # FIXED: Update Product State for next iteration
        curr_ps = next_cell * n_dfa + next_q


    return np.array(traj)


# ----------------------------
# Plotting on a 2D Map
# ----------------------------

def plot_sim(regions_dict, traj):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Plot Regions
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#e6ccff']
    c_idx = 0
    for label, bounds in regions_dict.items():
        if label == "None": continue
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                 linewidth=1, edgecolor='black', facecolor=colors[c_idx%5], alpha=0.5)
        ax.add_patch(rect)
        ax.text(x_min+0.1, y_min+0.1, label, fontsize=12, fontweight='bold')
        c_idx += 1

    # Plot Trajectory
    if len(traj) > 1:
        ax.plot(traj[:,0], traj[:,1], 'b.-', linewidth=1, markersize=3, label="Trajectory")
        ax.scatter(traj[0,0], traj[0,1], c='g', s=150, marker='*', label='Start', zorder=5)
        ax.scatter(traj[-1,0], traj[-1,1], c='r', s=150, marker='X', label='End', zorder=5)
        ax.legend()

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Symbolic Control: {len(traj)-1} Steps")
    plt.show()


# ----------------------------
# Plotting for GUI
# ----------------------------

def plot_sim_gui(regions_dict, traj):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Plot Regions
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#e6ccff']
    c_idx = 0
    for label, bounds in regions_dict.items():
        if label == "None": continue
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                 linewidth=1, edgecolor='black', facecolor=colors[c_idx%5], alpha=0.5)
        ax.add_patch(rect)
        ax.text(x_min+0.1, y_min+0.1, label, fontsize=12, fontweight='bold')
        c_idx += 1

    # Plot Trajectory
    if len(traj) > 1:
        ax.plot(traj[:,0], traj[:,1], 'b.-', linewidth=1, markersize=3, label="Trajectory")
        ax.scatter(traj[0,0], traj[0,1], c='g', s=150, marker='*', label='Start', zorder=5)
        ax.scatter(traj[-1,0], traj[-1,1], c='r', s=150, marker='X', label='End', zorder=5)
        ax.legend()

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Symbolic Control: {len(traj)-1} Steps")

    plt.close(fig)
    return fig