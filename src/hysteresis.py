import numpy as np
from numba import cuda


@cuda.jit
def hysterisis_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if x < input_array.shape[0] and y < input_array.shape[1]:
        if input_array[x, y] == 25:
            if (input_array[x - 1, y - 1] == 255 or
                    input_array[x - 1, y] == 255 or
                    input_array[x - 1, y + 1] == 255 or
                    input_array[x, y - 1] == 255 or
                    input_array[x, y + 1] == 255 or
                    input_array[x + 1, y - 1] == 255 or
                    input_array[x + 1, y] == 255 or
                    input_array[x + 1, y + 1] == 255):
                output_array[x, y] = 255
            else:
                output_array[x, y] = 0
        else:
            output_array[x, y] = input_array[x, y]
