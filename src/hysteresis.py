import numpy as np
from numba import cuda


@cuda.jit
def hysterisis_kernel(input_array, output_array):
    """
    Apply hysteresis thresholding to the input array.
    :param input_array: Input array (presumably an image)
    :param output_array: Output array to store the result
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_array
    if x < input_array.shape[0] and y < input_array.shape[1]:
        # Check if the current pixel value is 127
        if input_array[x, y] == 127:
            # Check if any neighboring pixel has a value of 255
            if (input_array[x - 1, y - 1] == 255 or
                    input_array[x - 1, y] == 255 or
                    input_array[x - 1, y + 1] == 255 or
                    input_array[x, y - 1] == 255 or
                    input_array[x, y + 1] == 255 or
                    input_array[x + 1, y - 1] == 255 or
                    input_array[x + 1, y] == 255 or
                    input_array[x + 1, y + 1] == 255):
                # Set the output pixel to 255 if any neighboring pixel has a value of 255
                output_array[x, y] = 255
            else:
                # Set the output pixel to 0 if none of the neighboring pixels have a value of 255
                output_array[x, y] = 0
        else:
            # Copy the input pixel value to the output if it is not equal to 127
            output_array[x, y] = input_array[x, y]
