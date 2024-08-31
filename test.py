import time


from numba import njit, prange
import numpy as np

# config.NUMBA_DIAGNOSTICS = 1  # Enable diagnostics


# @njit
@njit(parallel=True)
def compute_sum(arr):
    total = 0
    for i in prange(arr.shape[0]):
        total += arr[i]
    return total

# Generate a large random array
arr = np.random.rand(1000000)

# Measure the runtime
start_time = time.time()
result = compute_sum(arr)
end_time = time.time()

print(f"Result: {result}")
print(f"Runtime: {end_time - start_time} seconds")