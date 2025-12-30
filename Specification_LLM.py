import os
import google.generativeai as genai
import numpy as np
import json
from dotenv import load_dotenv


# ----------------------------
# LLM & Spec Parsing 
# ----------------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


def generate_spec_from_text(user_prompt):
    system_prompt = """
    You are the architect of a Symbolic Control Pipeline for a nonlinear mobile robot. You possess deep expertise in Formal Methods, Linear Temporal Logic (LTL), and Automata Theory.

  ### SYSTEM CONSTRAINTS
  1. **Workspace**: Continuous 2D plane: x ∈ [0, 10], y ∈ [0, 10], θ ∈ [-π, π].
  2. **Abstraction**: The continuous space is discretized into a finite set of labeled regions.
  3. **Implicit Label**: "None" represents free space (any coordinate not within a defined region).
  4. **Symbolic Controller**: The controller operates on the product state (Symbolic State × Automaton State).

  ### PROCESS OVERVIEW
  You must transform a User Goal into two distinct outputs:
  A. **Geometric Decomposition**: A set of rectangular regions.
  B. **Logic Synthesis**: A Deterministic Finite Automaton (DFA) governing the task.

  ### STEP 1: REGION ENGINEERING
  - **Logical Requests** (e.g., "Go to Kitchen"): Use user-provided names.
  - **Geometric Requests** (e.g., "Patrol boundaries", "Draw a Z"):
    1. Decompose the shape into an ordered sequence of rectangular waypoints (W0, W1, ...).
    2. Ensure regions are large enough to be reachable (min width/height > 0.5 units) but fit within bounds.
    3. Ensure connectivity: Consecutive regions in a geometric path should be close or overlapping to ensure reachability.

  ### STEP 2: DFA CONSTRUCTION RULES
  You must build a mathematically complete DFA.
  1. **Alphabet (Σ)**: The set of ALL defined region names + "None".
  2. **States (Q)**:
    - `q0`: Start state.
    - `q_accept`: The final success state (must include a self-loop on ALL inputs).
    - `q_trap`: The sink state for violations (must self-loop on ALL inputs). Define this state only in the presence of regions to avoid.
    - Intermediate states as required by the sequence.
  3. **Transition Function (δ: Q × Σ → Q)**:
    - **Completeness**: For EVERY state `q` and EVERY input `σ` in the alphabet, a transition must be defined.
    - **"None" Logic**: In non-accepting/non-trap states, input "None" must ALWAYS self-loop (δ(q, "None") = q). The robot needs to traverse free space to reach regions.
    - **Sequence Logic** ("Visit A then B"):
      - If in state looking for A: Input "A" moves to next state. Input "B" (premature arrival) typically self-loops (unless "Strict Ordering" is requested, then Trap).
    - **Safety/Avoidance** ("Avoid X"):
      - From ANY safe state, input "X" must transition to `q_trap`.

  ### STEP 3: REASONING & OUTPUT
  Before outputting JSON, perform a **Chain of Thought** analysis:
  1. List explicit regions and derived geometric waypoints.
  2. Define the Alphabet.
  3. Define the States needed to track progress.
  4. Draft the Transition Logic for standard progress, obstacles, and free space.

  ### OUTPUT FORMAT
  Return strictly valid JSON. Do not include markdown formatting (```json) inside the output object.

  {
    "reasoning": "Brief explanation of how the shape was decomposed and logic constructed.",
    "regions": {
      "RegionName": [[x_min, x_max], [y_min, y_max], [th_min, th_max]]
    },
    "dfa": {
      "states": ["q0", "q1", "q_trap", "q_accept"],
      "alphabet": ["RegionA", "RegionB", "Obstacle", "None"],
      "start_state": "q0",
      "accepting_states": ["q_accept"],
      "transitions": [
        {"from": "q0", "input": "RegionA", "to": "q1"},
        {"from": "q0", "input": "RegionB", "to": "q0"},
        {"from": "q0", "input": "Obstacle", "to": "q_trap"},
        {"from": "q0", "input": "None", "to": "q0"},
        ... (Repeat for ALL combinations of State x Alphabet) ...
      ]
    }
  }
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(f"{system_prompt}\n\nUser Goal: {user_prompt}")
        text = response.text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"LLM Error: {e}")
        return None
    

# ----------------------------
# JSON Parsing & Structure Building
# ----------------------------    
def parse_dfa_from_json(spec_json):
    """
    Parses JSON and builds the optimized Integer-based structures (Matrix/Maps).
    Returns 6 values.
    """
    regions = spec_json['regions']
    dfa_data = spec_json['dfa']
    a_states = dfa_data['states']
    a_init = dfa_data['start_state']
    a_accept = set(dfa_data['accepting_states'])

    # 1. Map State Names -> Integers
    state_map = {s: i for i, s in enumerate(a_states)}

    # 2. Map Region Labels -> Integers
    # "None" is always index 0
    region_names = ["None"] + list(regions.keys())
    r_map = {r: i for i, r in enumerate(region_names)}

    n_states = len(a_states)
    n_regions = len(region_names)

    # 3. Build DFA Transition Matrix [num_states, num_regions] -> next_state_idx
    dfa_matrix = np.zeros((n_states, n_regions), dtype=np.int32)
    for i in range(n_states):
        dfa_matrix[i, :] = i

    # Fill from JSON (Explicit Transitions)
    for t in dfa_data['transitions']:
        q_from_idx = state_map[t["from"]]
        r_input_idx = r_map[t["input"]]
        q_to_idx = state_map[t["to"]]
        dfa_matrix[q_from_idx, r_input_idx] = q_to_idx

    return regions, r_map, state_map, dfa_matrix, a_accept, a_init