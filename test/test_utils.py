import numpy as np


def generate_random_rgb_image(shape):
    return np.random.randint(0, 255, shape + (3,)).astype(np.uint8)


def generate_random_grayscale_image(shape):
    return np.random.randint(0, 255, shape).astype(np.uint8)
