import numpy as np
from numba import cuda


@cuda.jit
def grayscale_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if x < input_array.shape[0] and y < input_array.shape[1]:
        r, g, b = input_array[x, y]
        output_array[x, y] = 0.2989 * r + 0.5870 * g + 0.1140 * b
