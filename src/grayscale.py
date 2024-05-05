import numpy as np
from numba import cuda


@cuda.jit
def grayscale_kernel(input_array, output_array):
    """
    Convert RGB image to grayscale image
    :param input_array: Input RGB image array
    :param output_array: Output grayscale image array
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_array
    if x < input_array.shape[0] and y < input_array.shape[1]:
        # Extract the RGB values from the input_array
        r, g, b = input_array[x, y]

        # Compute the grayscale value using the luminance formula
        # Grayscale = 0.2989 * Red + 0.5870 * Green + 0.1140 * Blue
        grayscale_value = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Store the computed grayscale value in the output_array
        output_array[x, y] = grayscale_value
