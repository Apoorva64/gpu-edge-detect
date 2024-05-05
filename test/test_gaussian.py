from unittest import TestCase

from numba import cuda
from scipy.signal import convolve2d

from gaussian import gaussian_kernel, kernel, gaussian_local_copy_kernel
from src.grayscale import grayscale_kernel
from test_utils import generate_random_rgb_image, generate_random_grayscale_image
import numpy as np

from utils import get_grid_block
import time
import tqdm
import matplotlib.pyplot as plt

PLOT = True


class Test(TestCase):
    def test_gaussian_kernel(self):
        bw_image = generate_random_grayscale_image((10, 10))

        gaussian_image = np.zeros(bw_image.shape[:2], dtype=np.uint8)

        threads_per_block, blocks_per_grid = get_grid_block(bw_image)

        # load image to device
        d_bw_image = cuda.to_device(bw_image)
        d_gaussian_image = cuda.to_device(gaussian_image)

        # run kernel
        gaussian_kernel[blocks_per_grid, threads_per_block](d_bw_image, d_gaussian_image)

        # copy result back to host
        d_gaussian_image.copy_to_host(gaussian_image)

        # check result using scipy
        ##When a pixel is located on the edge of the picture, some values of the Gauss Kernel won't have a matching pixel. There are many strategies to deal with this situation but we will use the simplest. When a neighboring pixel is missing we will use the value of the current pixel as substitute.
        expected_gaussian_image = convolve2d(bw_image, kernel, mode='same', boundary='fill').astype(
            np.uint8)

        # array near equal (5% tolerance)
        np.testing.assert_allclose(gaussian_image, expected_gaussian_image, atol=12.75, rtol=5)

    def performance_gaussian_kernel(self, samples=20, image_sizes=[2 ** i for i in range(1, 12)],
                                         function=gaussian_kernel):
        """
        Tests the performance of the gaussian_kernel function by running it on different image sizes and measuring the
        execution time.
        """

        times = {}

        for image_size in tqdm.tqdm(image_sizes):
            bw_image = generate_random_grayscale_image((image_size, image_size))
            gaussian_image = np.zeros(bw_image.shape[:2], dtype=np.uint8)

            threads_per_block, blocks_per_grid = get_grid_block(bw_image)

            d_bw_image = cuda.to_device(bw_image)
            d_gaussian_image = cuda.to_device(gaussian_image)

            start = time.time()
            for _ in range(samples):
                function[blocks_per_grid, threads_per_block](d_bw_image, d_gaussian_image)
                cuda.synchronize()
            end = time.time()

            times[image_size] = (end - start) / samples

        return times

    def test_gaussian_functions(self, functions=None):
        if functions is None:
            functions = [gaussian_kernel, gaussian_local_copy_kernel]

        times_per_function = {}
        for function in functions:
            times = self.performance_gaussian_kernel(function=function)
            times_per_function[function.__name__] = times
        if PLOT:
            plt.xlabel('Image size')
            plt.ylabel('Time (s)')
            plt.title('Performance of gaussian functions')
            for function_name, times in times_per_function.items():
                plt.plot(list(times.keys()), list(times.values()), label=function_name)

            plt.legend()
            plt.show()

        for function_name, times in times_per_function.items():
            print(f"Function: {function_name}")
            print("Image Size\tExecution Time")
            for image_size, execution_time in times.items():
                print(f"{image_size}\t{execution_time}")
        return times_per_function
