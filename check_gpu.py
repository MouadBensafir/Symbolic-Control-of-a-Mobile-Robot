from numba import cuda
import numpy as np

try:
    print(f"CUDA Available: {cuda.is_available()}")
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"GPU Name: {device.name}")
        print(f"Compute Capability: {device.compute_capability}")
    else:
        print("CUDA is NOT available. Check your drivers.")
except Exception as e:
    print(f"Error: {e}")