import cmath
import math

import numpy as np
from numba import cuda


@cuda.jit
def non_max_suppression_kernel(input_array, output_array, angles):
    x, y = cuda.grid(2)
    if x >= input_array.shape[0] - 2 or y >= input_array.shape[1] - 2:
        return
    q = 255
    r = 255
    angle = angles[x, y]
    if angle < 0:
        angle += math.pi
    angle = math.degrees(angle)

    # angle 0
    if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
        q = input_array[x + 1, y]
        r = input_array[x - 1, y]

    # angle 45
    elif 22.5 <= angle < 67.5:
        q = input_array[x + 1, y - 1]
        r = input_array[x - 1, y + 1]

    # angle 90
    elif 67.5 <= angle < 112.5:
        q = input_array[x, y - 1]
        r = input_array[x, y + 1]

    # angle 135
    elif 112.5 <= angle < 157.5:
        q = input_array[x - 1, y - 1]
        r = input_array[x + 1, y + 1]

    if input_array[x, y] >= q and input_array[x, y] >= r:
        output_array[x, y] = input_array[x, y]
    else:
        output_array[x, y] = 0


