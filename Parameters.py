import numpy as np
import math

# ----------------------------
# Physics & Grid Parameters
# ----------------------------
Nx1, Nx2, Nx3 = 75, 75, 30
Nu1, Nu2 = 3, 5
tau = 1

x1_range = (0.0, 10.0)
x2_range = (0.0, 10.0)
x3_range = (-math.pi, math.pi)

u1_range = (0.25, 1.0)
u2_range = (-1.0, 1.0)

w1_bounds = (-0.05, 0.05)
w2_bounds = (-0.05, 0.05)
w3_bounds = (-0.05, 0.05)

# ----------------------------
# Discretization Setup
# ----------------------------
x1_bins = np.linspace(*x1_range, Nx1 + 1)
x2_bins = np.linspace(*x2_range, Nx2 + 1)
x3_bins = np.linspace(*x3_range, Nx3 + 1)

u1_vals = np.linspace(*u1_range, Nu1)
u2_vals = np.linspace(*u2_range, Nu2)
U_vals = [(u1_vals[i], u2_vals[j]) for i in range(Nu1) for j in range(Nu2)]

w_bounds = np.array([w1_bounds, w2_bounds, w3_bounds])
 
# Total discrete state cells
TOTAL_CELLS = Nx1 * Nx2 * Nx3

