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
    def secure_access(n, shape):
        if n < 0:
            return -n
        if n >= shape:
            diff = (shape - n) + 1
            return shape - diff
        return n

    x, y = cuda.grid(2)

    if input_image.shape[0] > x and input_image.shape[1] > y:
        output_image[x, y] = 0
        gx = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] +
              2 * input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] - 2 *
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] +
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        gy = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              2 * input_image[secure_access(x, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] - 2 *
              input_image[secure_access(x, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        # clamp values to 0-175
        output_image[x, y] = math.sqrt(gx ** 2 + gy ** 2)
        if output_image[x, y] > 175:
            output_image[x, y] = 175
        if output_image[x, y] < 0:
            output_image[x, y] = 0
@cuda.jit
def sobel_kernel_with_angle(input_image, output_image):
    """
    Applies the sobel operator to an image.
    :param input_image:
    :param output_image:
    :return:
    """
    def secure_access(n, shape):
        if n < 0:
            return -n
        if n >= shape:
            diff = (shape - n) + 1
            return shape - diff
        return n
    x, y = cuda.grid(2)

    if input_image.shape[0] > x and input_image.shape[1] > y:
        output_image[x, y] = 0
        gx = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] +
              2 * input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] - 2 *
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] +
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        gy = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              2 * input_image[secure_access(x, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] - 2 *
              input_image[secure_access(x, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        if gx < 0:
            gx = 0
        if gy < 0:
            gy = 0
        if gx > 175:
            gx = 175
        if gy > 175:
            gy = 175

        # clamp values to 0-175
        output_image[x, y] = int(math.ceil(math.sqrt(gx ** 2 + gy ** 2)))
