import numpy as np
from numba import cuda


@cuda.jit
def threshold_kernel(non_max_suppressed, output_array, low, high):
    """
    Apply thresholding to an image.
    :param non_max_suppressed: Input array after non-maximum suppression
    :param output_array: Output array to store the result
    :param low: Low threshold value
    :param high: High threshold value
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the non_max_suppressed array
    if x < non_max_suppressed.shape[0] and y < non_max_suppressed.shape[1]:
        # Apply thresholding based on the values in non_max_suppressed array
        if non_max_suppressed[x, y] >= high:
            # If the value is greater than or equal to the high threshold, set the output pixel to 255 (white)
            output_array[x, y] = 255
        elif non_max_suppressed[x, y] < low:
            # If the value is less than the low threshold, set the output pixel to 0 (black)
            output_array[x, y] = 0
        else:
            # If the value is between the low and high thresholds, set the output pixel to 127 (gray)
            output_array[x, y] = 127
