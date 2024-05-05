from numba import cuda

DEFAULT_THREADS_PER_BLOCK = cuda.get_current_device().WARP_SIZE, cuda.get_current_device().WARP_SIZE


def get_grid_block(image):
    """
    Calculate the grid and block dimensions for CUDA kernel execution based on the input image size.
    :param image: Input image array
    :return: Tuple containing threads per block and blocks per grid
    """
    # Define the default number of threads per block
    threads_per_block = DEFAULT_THREADS_PER_BLOCK

    # Calculate the number of blocks per grid in the x and y directions
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]

    # Combine the number of blocks per grid in both directions into a tuple
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Return the threads per block and blocks per grid as a tuple
    return threads_per_block, blocks_per_grid
