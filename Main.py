from Parameters import TOTAL_CELLS, U_vals, x1_range, x2_range
from Labels import precompute_labels_parallel
from Abstraction import build_transition_table_parallel
from Specification_LLM import generate_spec_from_text, parse_dfa_from_json
from Safety import compute_safety_implicit_parallel
from Reachability import compute_optimal_reachability
from Controller import synthesize_optimal_controller
from Plotting import simulate, plot_sim
from Animation import animate
import numpy as np
    
    
# ----------------------------
# 1. Get User Goal using LLM
# ----------------------------
print("="*60)
print("🤖  Welcome to the Autonomous Controller Synthesis System!")
print("="*60)

user_goal = input("\n🔹 State your goal: ")

print(f"\n🗣️ Goal recorded : {user_goal}")
print("🤖 Calling LLM...\n")
spec_json = generate_spec_from_text(user_goal)

# Stop pipeline if LLM failed
if spec_json is None:
    print("LLM Failed to generate a valid specification. Exiting.")
    exit(1)
    
    
# ----------------------------
# 2. Build Transition Table
# ----------------------------
T = build_transition_table_parallel()
print(f"✅ Physics Ready. Total Transitions: {len(T)}")

 
# ----------------------------
# 3. Parse Specification
# ----------------------------
regions, r_map, s_map, dfa_matrix, a_accept, a_init = parse_dfa_from_json(spec_json)

# This uses Parallelism to label all cells instantly
label_map = precompute_labels_parallel(regions, r_map)

# 4. Setup Synthesis
n_cells = TOTAL_CELLS
n_dfa = len(s_map)
n_u = len(U_vals)

# Identify Trap States (Indices in DFA)
bad_q_indices = [s_map[q] for q in s_map if "trap" in q.lower()]

# Identify Target States (Indices in DFA)
target_q_indices = [s_map[q] for q in a_accept]

print("\n--- SYNTHESIS ---")

# Safety Game
safe_mask = compute_safety_implicit_parallel(
    T, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u
)
print(f"   Safe States: {np.sum(safe_mask)}")

# Reachability Game 
target_mask = np.zeros(n_cells * n_dfa, dtype=bool)
for q_idx in target_q_indices:
    # Set True for every cell at this Q index
    target_mask[q_idx::n_dfa] = True

V, win_mask = compute_optimal_reachability(
    safe_mask, target_mask, T, dfa_matrix, label_map, n_dfa, n_u
)
print(f"   Winning States: {np.sum(win_mask)}")


# ----------------------------
# Synthesize Controller  
# ----------------------------
if np.sum(win_mask) > 0:
    controller = synthesize_optimal_controller(V, win_mask, T, dfa_matrix, label_map, n_dfa, n_u)
    print(f"✅ Controller Synthesized. Size: {len(controller)}")
else:
    print("❌ Winning Set is empty! Cannot synthesize controller.")
    
    
# ----------------------------
# Run Simulation and Plot 
# ----------------------------
traj = []
while True:
    print("\n-------------------------------------------")
    print("📍  Robot Initial Position")
    print("-------------------------------------------")

    try:
        x_init, y_init = map(int, input("Enter initial position (x y): ").split())
    except ValueError:
        print("⚠️ Please enter two integers separated by space.")
        continue

    start_x = np.array([x_init, y_init, 0])
    print("\n✔ Initial state recorded:", start_x)
    print("-------------------------------------------")

    # Run trajectory
    traj = simulate(start_x, s_map[a_init], controller, dfa_matrix, label_map, n_dfa, U_vals, regions, s_map, a_accept)
    plot_sim(regions, traj)

    # Ask user if they want another run
    choice = input("\n🔁 Run another trajectory with a new initial point? (y/n): ").strip().lower()
    if choice not in ("y", "yes"):
        print("\n Exiting trajectory simulation.")
        break


# ----------------------------
# Pybullet Simulation
# ----------------------------
try:
    print("\n Taking last recorded trajectory for animation...")
    animate(traj, regions, fps=60)
except Exception as e:
    print(f"Animation failed: {e}")