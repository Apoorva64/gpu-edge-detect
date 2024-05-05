import numpy as np
from numba import cuda


kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]], np.float32) / 256


@cuda.jit
def gaussian_kernel(input_array, output_array):
    """
    Apply a Gaussian filter kernel to the input array.
    :param input_array: Input array (presumably an image)
    :param output_array: Output array to store the result
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are out of bounds
    if x >= input_array.shape[0] or y >= input_array.shape[1]:
        return

    # Iterate over the elements of the kernel
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            # Calculate the indices of the corresponding pixel in the input_array
            input_x = x + i - 2
            input_y = y + j - 2

            # Check if the calculated indices are out of bounds
            if input_x < 0 or input_x >= input_array.shape[0] or input_y < 0 or input_y >= input_array.shape[1]:
                # Use the same pixel value if the kernel is out of bounds
                output_array[x, y] += input_array[x, y] * kernel[i, j]
            else:
                # Apply the Gaussian filter by multiplying the input pixel value with the corresponding kernel value
                output_array[x, y] += input_array[input_x, input_y] * kernel[i, j]


@cuda.jit
def gaussian_local_copy_kernel(input_array, output_array):
    """
    Apply a Gaussian filter kernel to the input array using shared memory for kernel access.
    :param input_array: Input array (presumably an image)
    :param output_array: Output array to store the result
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are out of bounds
    if x >= input_array.shape[0] or y >= input_array.shape[1]:
        return

    # Define a shared memory array to hold the kernel
    kernel_local = cuda.shared.array((5, 5), dtype=np.float32)

    # Copy the global kernel to shared memory
    for i in range(kernel_local.shape[0]):
        for j in range(kernel_local.shape[1]):
            kernel_local[i, j] = kernel[i, j]

    # Synchronize threads to ensure all threads have copied the kernel before proceeding
    cuda.syncthreads()

    # Apply the Gaussian filter using the copied kernel from shared memory
    for i in range(kernel_local.shape[0]):
        for j in range(kernel_local.shape[1]):
            # Calculate the indices of the corresponding pixel in the input_array
            input_x = x + i - 2
            input_y = y + j - 2

            # Check if the calculated indices are out of bounds
            if input_x < 0 or input_x >= input_array.shape[0] or input_y < 0 or input_y >= input_array.shape[1]:
                # Use the same pixel value if the kernel is out of bounds
                output_array[x, y] += input_array[x, y] * kernel_local[i, j]
            else:
                # Apply the Gaussian filter by multiplying the input pixel value with the corresponding kernel value
                output_array[x, y] += input_array[input_x, input_y] * kernel_local[i, j]
