import cmath
import math

import numpy as np
from numba import cuda


@cuda.jit
def sobel_kernel(input_image, output_image):
    """
    Applies the sobel operator to an image.
    :param input_image:
    :param output_image:
    :return:
    """
    x, y = cuda.grid(2)

    if input_image.shape[0] - 2 > x > 0 and input_image.shape[1] - 2 > y > 0:
        output_image[x + 1, y + 1] = 0
        gx = (input_image[x, y] - input_image[x + 2, y] +
              2 * input_image[x, y + 1] - 2 * input_image[x + 2, y + 1] +
              input_image[x, y + 2] - input_image[x + 2, y + 2])

        gy = (input_image[x, y] - input_image[x, y + 2] +
              2 * input_image[x + 1, y] - 2 * input_image[x + 1, y + 2] +
              input_image[x + 2, y] - input_image[x + 2, y + 2])

        # clamp values to 0-175
        output_image[x + 1, y + 1] = math.sqrt(gx ** 2 + gy ** 2)
        if output_image[x + 1, y + 1] > 175:
            output_image[x + 1, y + 1] = 175
        if output_image[x + 1, y + 1] < 0:
            output_image[x + 1, y + 1] = 0
        # arctan2 for angles


@cuda.jit
def sobel_kernel_with_angle(input_image, output_image, angles):
    """
    Applies the sobel operator to an image.
    :param input_image:
    :param output_image:
    :return:
    """
    x, y = cuda.grid(2)

    if input_image.shape[0] - 2 > x > 0 and input_image.shape[1] - 2 > y > 0:
        output_image[x + 1, y + 1] = 0
        gx = (input_image[x, y] - input_image[x + 2, y] +
              2 * input_image[x, y + 1] - 2 * input_image[x + 2, y + 1] +
              input_image[x, y + 2] - input_image[x + 2, y + 2])

        gy = (input_image[x, y] - input_image[x, y + 2] +
              2 * input_image[x + 1, y] - 2 * input_image[x + 1, y + 2] +
              input_image[x + 2, y] - input_image[x + 2, y + 2])

        # clamp values to 0-175
        output_image[x + 1, y + 1] = math.sqrt(gx ** 2 + gy ** 2)
        if output_image[x + 1, y + 1] > 175:
            output_image[x + 1, y + 1] = 175
        if output_image[x + 1, y + 1] < 0:
            output_image[x + 1, y + 1] = 0
        # arctan2 for angles
        angles[x + 1, y + 1] = cmath.phase(complex(gx, gy))
