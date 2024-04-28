from unittest import TestCase

from numba import cuda

from src.grayscale import grayscale_kernel
from test_utils import generate_random_rgb_image
import numpy as np

from utils import get_grid_block
import time
import tqdm
import matplotlib.pyplot as plt

PLOT = True


class Test(TestCase):
    def test_grayscale_kernel(self):
        rgb_image = generate_random_rgb_image((10, 10))

        grayscale_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

        threads_per_block, blocks_per_grid = get_grid_block(rgb_image)

        # load image to device
        d_rgb_image = cuda.to_device(rgb_image)
        d_grayscale_image = cuda.to_device(grayscale_image)

        # run kernel
        grayscale_kernel[blocks_per_grid, threads_per_block](d_rgb_image, d_grayscale_image)

        # copy result back to host
        d_grayscale_image.copy_to_host(grayscale_image)

        # check result
        expected_grayscale_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        np.testing.assert_array_equal(grayscale_image, expected_grayscale_image)

    def test_performance_grayscale_kernel(self):
        """
        Tests the performance of the grayscale_kernel function by running it on different image sizes and measuring the
        execution time.
        """

        times = {}
        samples = 100
        image_sizes = [2 ** i for i in range(1, 14)]

        for image_size in tqdm.tqdm(image_sizes):
            rgb_image = generate_random_rgb_image((image_size, image_size))
            grayscale_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

            threads_per_block, blocks_per_grid = get_grid_block(rgb_image)

            d_rgb_image = cuda.to_device(rgb_image)
            d_grayscale_image = cuda.to_device(grayscale_image)

            start = time.time()
            for _ in range(samples):
                grayscale_kernel[blocks_per_grid, threads_per_block](d_rgb_image, d_grayscale_image)
                cuda.synchronize()
            end = time.time()

            times[image_size] = (end - start) / samples

        # print results in a table
        print("Image Size\tExecution Time")
        for image_size, execution_time in times.items():
            print(f"{image_size}\t{execution_time}")

        if PLOT:
            plt.plot(list(times.keys()), list(times.values()))
            plt.xlabel("Image Size")
            plt.ylabel("Execution Time")
            plt.title("Performance of grayscale_kernel")
            plt.show()
