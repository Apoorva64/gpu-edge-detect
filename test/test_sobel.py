import math
from unittest import TestCase

from numba import cuda

from src.sobel import sobel_kernel
from test_utils import generate_random_rgb_image, generate_random_grayscale_image
import numpy as np

from utils import get_grid_block
import time
import tqdm
import matplotlib.pyplot as plt

PLOT = True


class Test(TestCase):
    def test_sobel_kernel(self):
        bw_image = generate_random_grayscale_image((10, 10))

        sobel_image = np.zeros(bw_image.shape[:2], dtype=np.uint8)

        threads_per_block, blocks_per_grid = get_grid_block(bw_image)

        d_bw_image = cuda.to_device(bw_image)
        d_sobel_image = cuda.to_device(sobel_image)

        sobel_kernel[blocks_per_grid, threads_per_block](d_bw_image, d_sobel_image)

        d_sobel_image.copy_to_host(sobel_image)

    def test_performance_sobel_kernel(self):
        """
        Tests the performance of the sobel_kernel function by running it on different image sizes and measuring the
        execution time.
        """

        times = {}
        samples = 100
        image_sizes = [2 ** i for i in range(1, 14)]

        for image_size in tqdm.tqdm(image_sizes):
            bw_image = generate_random_grayscale_image((image_size, image_size))
            sobel_image = np.zeros(bw_image.shape[:2], dtype=np.uint8)

            threads_per_block, blocks_per_grid = get_grid_block(bw_image)

            d_bw_image = cuda.to_device(bw_image)
            d_sobel_image = cuda.to_device(sobel_image)

            start = time.time()
            for _ in range(samples):
                sobel_kernel[blocks_per_grid, threads_per_block](d_bw_image, d_sobel_image)
            cuda.synchronize()
            times[image_size] = (time.time() - start) / samples

        if PLOT:
            plt.plot(list(times.keys()), list(times.values()))
            plt.xlabel('Image size')
            plt.ylabel('Time (s)')
            plt.title('Performance of sobel_kernel')
            plt.show()
