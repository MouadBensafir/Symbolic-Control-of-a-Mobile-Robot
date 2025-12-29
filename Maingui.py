import flet as ft
import threading
import os
import numpy as np
import time

from Parameters import TOTAL_CELLS, U_vals
from Abstraction import build_transition_table_parallel
from Specification_LLM import generate_spec_from_text, parse_dfa_from_json
from Labels import precompute_labels_parallel
from Safety import compute_safety_implicit_parallel
from Reachability import compute_optimal_reachability
from Controller import synthesize_optimal_controller
from Plotting import simulate, plot_sim_gui     
from Animation import animate_gui       

# from flet.matplotlib_chart import MatplotlibChart      

def create_plot_card(regions, traj):
    fig = plot_sim_gui(regions, traj)
    
    chart = ft.MatplotlibChart(fig, expand=True, transparent=False)

    return ft.Container(
        content=chart,
        padding=10,
        border=ft.border.all(1, "#3f3f46"),
        border_radius=10,
        bgcolor ="white",
        expand=True
    )


T = None  # Global Transition Table

def main(page: ft.Page):
    page.title = "SymBot-LLM"    
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#09090b"
    page.window_state = "maximized"
    page.padding = 40
    page.update()

    # Results placeholder:
    results_column = ft.Column([], expand=True)
    

    # --- UI COMPONENT: LOADING VIEW ---
    loading_ring = ft.ProgressRing(color="white", width=50, height=50)
    loading_text = ft.Column([
        ft.Text("Building Transition Table...", color="#a1a1aa", size=16),
        ft.Text("300,000 states are being processed", color="#71717a", size=12)
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    loading_content = ft.Column([loading_ring, loading_text],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER
    )

    loading_view = ft.Container(
        content=loading_content,
        alignment=ft.Alignment.CENTER,
        expand=True,
        visible=True
    )

    current_synthesis_log = ft.Text("", color="#a1a1aa", size=16)

    synthesis_text = ft.Column([
        current_synthesis_log
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)


    # --- UI COMPONENT: DASHBOARD VIEW ---
    input_x = ft.TextField(label="X", width=100, value="0", bgcolor="#18181b", border_color="#27272a")
    input_y = ft.TextField(label="Y", width=100, value="0", bgcolor="#18181b", border_color="#27272a")
    
    prompt_input = ft.TextField(
        label="Enter User Goal",
        hint_text="e.g., Go to the top right avoiding region C",
        bgcolor="#18181b", 
        border_color="#27272a",
        expand=True
    )

    def run_synthesis(e):
            synthesis_log = ft.Column([
                loading_ring, synthesis_text], 
                horizontal_alignment="center",
                alignment="center",
                expand=True,)

            synthesis_content = ft.Container(
                content=synthesis_log,
                alignment=ft.Alignment.CENTER,
                expand=True,
            )

            results_column.controls.clear() # Clear previous results
            results_column.controls.append(synthesis_content)

            prompt = prompt_input.value
            if not prompt:
                return
            
            # Show specific loading state for synthesis
            btn_synthesize.text = "Synthesizing..."
            btn_synthesize.disabled = True
            page.update()

            # Generation spec DFA from LLM

            current_synthesis_log.value = "Generating specification Automaton.."
            page.update()
            spec_json = generate_spec_from_text(prompt)


            # Stop pipeline if LLM failed
            if spec_json is None:
                current_synthesis_log.value = "LLM Failed to generate a valid specification. Exiting.."
                page.update()
                exit(1)

            # ----------------------------
            # 3. Parse Specification
            # ----------------------------
            regions, r_map, s_map, dfa_matrix, a_accept, a_init = parse_dfa_from_json(spec_json)

            # This uses Parallelism to label all cells instantly
            current_synthesis_log.value = "Precomputing Labels for all states..."
            page.update()
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
            current_synthesis_log.value = "Computing Safety.."
            page.update()
            safe_mask = compute_safety_implicit_parallel(
                T, dfa_matrix, label_map, bad_q_indices, n_cells, n_dfa, n_u
            )
            print(f"   Safe States: {np.sum(safe_mask)}")

            # Reachability Game 
            current_synthesis_log.value = "Computing Reachability.."
            page.update()
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

                current_synthesis_log.value = "Synthesizing Controller..."
                page.update()
                controller = synthesize_optimal_controller(V, win_mask, T, dfa_matrix, label_map, n_dfa, n_u)

                def simulate_and_display(e):
                    results_column.controls.clear()
                    current_synthesis_log.value = "Preparing Simulation..."
                    results_column.controls.append(synthesis_content)
                    page.update()

                    try:
                            x0 = float(input_x.value)
                            y0 = float(input_y.value)
                    except ValueError:
                        x0, y0 = 0.0, 0.0  # Default position if invalid input 

                    start_pos = np.array([x0, y0,0])
                    current_synthesis_log.value = "Simulating Trajectory..."
                    page.update()
                    traj = simulate(start_pos, s_map[a_init], controller, dfa_matrix, label_map, n_dfa, U_vals, regions, s_map, a_accept)

                    # Create plot card and add to results
                    current_synthesis_log.value = "Creating Plot..."
                    page.update()
                    plot_card = create_plot_card(regions, traj)
                    page.update()

                    # Animate the simulation
                    current_synthesis_log.value = "Creating Animation..."
                    page.update()
                    save_path = f"animation{int(x0)}_{int(y0)}.gif"
                    animation_path = animate_gui(traj,regions, save_path)  

                    # Animation Card
                    anim_card = ft.Container(
                        content=ft.Image(src=save_path, fit=ft.ImageFit.CONTAIN),
                        padding=10,
                        border=ft.border.all(1, "#3f3f46"),
                        border_radius=10,
                        bgcolor="#18181b",
                        expand=True
                    )

                    current_synthesis_log.value = "Synthesis Complete."
                    synthesis_log = ft.Column([
                        synthesis_text], 
                        horizontal_alignment="center",
                        alignment="center",
                        expand=True,)
                    page.update()

                    
                    results_row = ft.Row(
                        controls=[plot_card, anim_card], 
                        expand=True, 
                        alignment=ft.MainAxisAlignment.CENTER
                    )
                    results_column.controls.clear()
                    results_column.controls.append(simulation_input)
                    results_column.controls.append(results_row)
                    page.update()

                position_input = ft.Row([ft.Text("Start Position:", color="#a1a1aa"), input_x, input_y], spacing=20)
                
                simulation_button = ft.ElevatedButton(
                    "Launch Simulation",
                    on_click=simulate_and_display,
                    color="black",
                    bgcolor="white",
                    height=50,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
                )
                simulation_input = ft.Row([position_input, simulation_button], alignment=ft.MainAxisAlignment.CENTER, spacing=30)

                results_column.controls.clear()
                results_column.controls.append(simulation_input)
                page.update()

            else:
                current_synthesis_log.value = "Winning Set is empty! Cannot synthesize controller."
                page.update()
                exit(1)

            # Reset button
            btn_synthesize.text = "Synthesize Controller"
            btn_synthesize.disabled = False
            page.update()

    btn_synthesize = ft.ElevatedButton(
        "Synthesize Controller",
        on_click=run_synthesis, # Ensure this points to your function
        color="black",
        bgcolor="white",
        height=50,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
    )

    dashboard_view = ft.Column([
        ft.Text("SymBot-LLM", size=32, weight="bold", color="white"),
        ft.Text("Symbolic Controller Synthesis", size=16, color="#a1a1aa"),
        ft.Row([prompt_input, btn_synthesize]),
        ft.Divider(color="#27272a", height=40),
        results_column 
    ], visible=False) # Strictly hidden until physics is ready


    # --- ADD TO PAGE ONCE ---
    # By adding both at once, Flet manages the visibility toggle correctly
    page.add(loading_view, dashboard_view)

    # ---------------------------------------------------------
    # Background Logic: Build Physics
    # ---------------------------------------------------------
    def build_physics():
        global T
        # Call your heavy function
        T = build_transition_table_parallel()
        
        # When done, switch UI
        loading_view.visible = False
        dashboard_view.visible = True
        page.update()

    # Start the thread
    threading.Thread(target=build_physics, daemon=True).start()


if __name__ == "__main__":
    ft.app(target=main)