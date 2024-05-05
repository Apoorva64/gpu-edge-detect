import numpy as np
from numba import cuda


@cuda.jit
def threshold_kernel(non_max_suppressed, output_array, low, high):
    x, y = cuda.grid(2)
    if x < non_max_suppressed.shape[0] and y < non_max_suppressed.shape[1]:
        if non_max_suppressed[x, y] >= high:
            output_array[x, y] = 255
        elif non_max_suppressed[x, y] < low:
            output_array[x, y] = 0
        else:
            output_array[x, y] = 25
