from Parameters import Nx1, Nx2, Nx3, tau, x1_bins, x2_bins, x3_bins
import numpy as np
import math

# ----------------------------
# Integer Indexing (for RAM Optimization)
# ----------------------------
STRIDE_X1 = Nx2 * Nx3
STRIDE_X2 = Nx3
TOTAL_CELLS = Nx1 * Nx2 * Nx3

def cell_to_int(i, j, k):
    """(i,j,k) -> flat_integer"""
    return i * STRIDE_X1 + j * STRIDE_X2 + k

def int_to_cell(idx):
    """flat_integer -> (i,j,k)"""
    i = idx // STRIDE_X1
    rem = idx % STRIDE_X1
    j = rem // STRIDE_X2
    k = rem % STRIDE_X2
    return (i, j, k)


# ----------------------------
# Dynamics Function (the Model)
# ----------------------------
def f_cont(x, u, w, tau=tau):
    v, omega = u
    wx, wy, wt = w
    x1 = x[0] + tau * (v * math.cos(x[2]) + wx)
    x2 = x[1] + tau * (v * math.sin(x[2]) + wy)
    x3 = x[2] + tau * (omega + wt)
    x3 = (x3 + math.pi) % (2 * math.pi) - math.pi
    return np.array((x1, x2, x3))


# ----------------------------
# Discretization Funtion
# ----------------------------
def q(x):
    i = np.clip(np.digitize(x[0], x1_bins) - 1, 0, Nx1 - 1)
    j = np.clip(np.digitize(x[1], x2_bins) - 1, 0, Nx2 - 1)
    k = np.clip(np.digitize(x[2], x3_bins) - 1, 0, Nx3 - 1)
    return (int(i), int(j), int(k))


# ----------------------------
# Contretization Function
# ----------------------------
def p(cell):
    i, j, k = cell
    return np.array(
        (
            0.5 * (x1_bins[i] + x1_bins[i + 1]),
            0.5 * (x2_bins[j] + x2_bins[j + 1]),
            0.5 * (x3_bins[k] + x3_bins[k + 1]),
        )
    )