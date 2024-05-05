import cmath
import math

import numpy as np
from numba import cuda


@cuda.jit
def sobel_kernel(input_image, output_image):
    """
    Applies the Sobel operator to an image.
    :param input_image: Input image array
    :param output_image: Output image array to store the result
    :return: None
    """
    def secure_access(n, shape):
        """
        Helper function to access array indices securely.
        :param n: Index
        :param shape: Shape of the array
        :return: Valid index within array bounds
        """
        # If index is negative, return its absolute value
        if n < 0:
            return -n
        # If index is greater than or equal to array shape, return a valid index within array bounds
        if n >= shape:
            diff = (shape - n) + 1
            return shape - diff
        # Otherwise, return the index itself
        return n

    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_image
    if input_image.shape[0] > x and input_image.shape[1] > y:
        # Calculate the horizontal gradient (gx)
        gx = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] +
              2 * input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] - 2 *
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] +
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        # Calculate the vertical gradient (gy)
        gy = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              2 * input_image[secure_access(x, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] - 2 *
              input_image[secure_access(x, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        # Compute the gradient magnitude
        gradient_magnitude = math.sqrt(gx ** 2 + gy ** 2)

        # Clamp values to the range [0, 175]
        if gradient_magnitude > 175:
            gradient_magnitude = 175
        if gradient_magnitude < 0:
            gradient_magnitude = 0

        # Store the gradient magnitude in the output_image
        output_image[x, y] = gradient_magnitude
