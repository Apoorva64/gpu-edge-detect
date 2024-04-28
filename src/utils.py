from numba import cuda


def get_grid_block(image):
    gpu = cuda.get_current_device()
    threads_per_block = (gpu.WARP_SIZE, gpu.WARP_SIZE)
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return threads_per_block, blocks_per_grid
