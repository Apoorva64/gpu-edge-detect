import numpy as np
from numba import cuda

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]]) / 256


@cuda.jit
def gaussian_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if x >= input_array.shape[0] or y >= input_array.shape[1]:
        return
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            # if the kernel is out of bounds, we use the same pixel value
            if x + i - 2 < 0 or x + i - 2 >= input_array.shape[0] or y + j - 2 < 0 or y + j - 2 >= input_array.shape[1]:
                output_array[x, y] += input_array[x, y] * kernel[i, j]
            else:
                output_array[x, y] += input_array[x + i - 2, y + j - 2] * kernel[i, j]


@cuda.jit
def gaussian_local_copy_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if x >= input_array.shape[0] or y >= input_array.shape[1]:
        return
    kernel_local = cuda.shared.array((5, 5), dtype=np.float32)
    for i in range(kernel_local.shape[0]):
        for j in range(kernel_local.shape[1]):
            kernel_local[i, j] = kernel[i, j]

    cuda.syncthreads()
    for i in range(kernel_local.shape[0]):
        for j in range(kernel_local.shape[1]):
            # if the kernel is out of bounds, we use the same pixel value
            if x + i - 2 < 0 or x + i - 2 >= input_array.shape[0] or y + j - 2 < 0 or y + j - 2 >= input_array.shape[1]:
                output_array[x, y] += input_array[x, y] * kernel_local[i, j]
            else:
                output_array[x, y] += input_array[x + i - 2, y + j - 2] * kernel_local[i, j]
