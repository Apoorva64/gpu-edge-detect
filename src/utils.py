from numba import cuda

DEFAULT_THREADS_PER_BLOCK = cuda.get_current_device().WARP_SIZE, cuda.get_current_device().WARP_SIZE

def get_grid_block(image):
    threads_per_block = DEFAULT_THREADS_PER_BLOCK
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return threads_per_block, blocks_per_grid
