from joblib import Parallel, delayed
from Utils import int_to_cell
from Parameters import x1_bins, x2_bins, x3_bins, TOTAL_CELLS
import numpy as np
import multiprocessing


# ----------------------------
# Label Function (Geometric Containment)
# ----------------------------

def precompute_labels_parallel(regions_dict, r_map):
    """
    Returns array of shape (TOTAL_CELLS,) where val = region_index.
    Uses parallel workers to check geometric containment for all 300k cells.
    """
    num_cores = multiprocessing.cpu_count()

    def check_chunk(indices):
        local_labels = np.zeros(len(indices), dtype=np.int8)

        # Unpack regions for speed
        r_keys = list(regions_dict.keys())
        r_bounds = [regions_dict[k] for k in r_keys]
        r_indices = [r_map[k] for k in r_keys]

        for k, cell_idx in enumerate(indices):
            i, j, m = int_to_cell(cell_idx)

            # Cell boundaries
            cx_lo, cx_hi = x1_bins[i], x1_bins[i+1]
            cy_lo, cy_hi = x2_bins[j], x2_bins[j+1]
            cth_lo, cth_hi = x3_bins[m], x3_bins[m+1]

            lbl_idx = 0 # Default "None"

            # Check strict containment against all regions
            for r_i, bounds in enumerate(r_bounds):
                rx, ry, rth = bounds
                # Check: Region contains Cell
                if (rx[0] <= cx_lo and cx_hi <= rx[1] and
                    ry[0] <= cy_lo and cy_hi <= ry[1] and
                    rth[0] <= cth_lo and cth_hi <= rth[1]):
                    lbl_idx = r_indices[r_i]
                    break

            local_labels[k] = lbl_idx
        return local_labels

    all_indices = np.arange(TOTAL_CELLS)
    chunk_size = len(all_indices) // num_cores + 1
    chunks = [all_indices[i:i + chunk_size] for i in range(0, len(all_indices), chunk_size)]

    print("Pre-computing Geometric Labels...")
    results = Parallel(n_jobs=num_cores)(delayed(check_chunk)(chunk) for chunk in chunks)

    return np.concatenate(results)