from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

LOW_THRESHOLD = 51
HIGH_THRESHOLD = 102

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

@cuda.jit
def grayscale_kernel(input_array, output_array):
    """
    Convert RGB image to grayscale image
    :param input_array: Input RGB image array
    :param output_array: Output grayscale image array
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_array
    if x < input_array.shape[0] and y < input_array.shape[1]:
        # Extract the RGB values from the input_array
        r, g, b = input_array[x, y]

        # Compute the grayscale value using the luminance formula
        # Grayscale = 0.2989 * Red + 0.5870 * Green + 0.1140 * Blue
        grayscale_value = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Store the computed grayscale value in the output_array
        output_array[x, y] = grayscale_value

@cuda.jit
def hysterisis_kernel(input_array, output_array):
    """
    Apply hysteresis thresholding to the input array.
    :param input_array: Input array (presumably an image)
    :param output_array: Output array to store the result
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_array
    if x < input_array.shape[0] and y < input_array.shape[1]:
        # Check if the current pixel value is 127
        if input_array[x, y] == 127:
            # Check if any neighboring pixel has a value of 255
            if (input_array[x - 1, y - 1] == 255 or
                    input_array[x - 1, y] == 255 or
                    input_array[x - 1, y + 1] == 255 or
                    input_array[x, y - 1] == 255 or
                    input_array[x, y + 1] == 255 or
                    input_array[x + 1, y - 1] == 255 or
                    input_array[x + 1, y] == 255 or
                    input_array[x + 1, y + 1] == 255):
                # Set the output pixel to 255 if any neighboring pixel has a value of 255
                output_array[x, y] = 255
            else:
                # Set the output pixel to 0 if none of the neighboring pixels have a value of 255
                output_array[x, y] = 0
        else:
            # Copy the input pixel value to the output if it is not equal to 127
            output_array[x, y] = input_array[x, y]

@cuda.jit
def sobel_kernel(input_image, output_image):
    """
    Applies the Sobel operator to an image.
    :param input_image: Input image array
    :param output_image: Output image array to store the result
    :return: None
    """
    def secure_access(n, shape):
        """
        Helper function to access array indices securely.
        :param n: Index
        :param shape: Shape of the array
        :return: Valid index within array bounds
        """
        # If index is negative, return its absolute value
        if n < 0:
            return -n
        # If index is greater than or equal to array shape, return a valid index within array bounds
        if n >= shape:
            diff = (shape - n) + 1
            return shape - diff
        # Otherwise, return the index itself
        return n

    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the input_image
    if input_image.shape[0] > x and input_image.shape[1] > y:
        # Calculate the horizontal gradient (gx)
        gx = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] +
              2 * input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] - 2 *
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y, input_image.shape[1])] +
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        # Calculate the vertical gradient (gy)
        gy = (input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x - 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              2 * input_image[secure_access(x, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] - 2 *
              input_image[secure_access(x, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])] +
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y - 1, input_image.shape[1])] -
              input_image[secure_access(x + 1, input_image.shape[0]), secure_access(y + 1, input_image.shape[1])])

        # Compute the gradient magnitude
        gradient_magnitude = math.sqrt(gx ** 2 + gy ** 2)

        # Clamp values to the range [0, 175]
        if gradient_magnitude > 175:
            gradient_magnitude = 175
        if gradient_magnitude < 0:
            gradient_magnitude = 0

        # Store the gradient magnitude in the output_image
        output_image[x, y] = gradient_magnitude

@cuda.jit
def threshold_kernel(non_max_suppressed, output_array, low, high):
    """
    Apply thresholding to an image.
    :param non_max_suppressed: Input array after non-maximum suppression
    :param output_array: Output array to store the result
    :param low: Low threshold value
    :param high: High threshold value
    :return: None
    """
    # Get the thread indices in a 2D grid
    x, y = cuda.grid(2)

    # Check if the thread indices are within the bounds of the non_max_suppressed array
    if x < non_max_suppressed.shape[0] and y < non_max_suppressed.shape[1]:
        # Apply thresholding based on the values in non_max_suppressed array
        if non_max_suppressed[x, y] >= high:
            # If the value is greater than or equal to the high threshold, set the output pixel to 255 (white)
            output_array[x, y] = 255
        elif non_max_suppressed[x, y] < low:
            # If the value is less than the low threshold, set the output pixel to 0 (black)
            output_array[x, y] = 0
        else:
            # If the value is between the low and high thresholds, set the output pixel to 127 (gray)
            output_array[x, y] = 127

def main(args):
    # Read the image

    image = Image.open(args.inputImage, 'r')
    # if image is rgba, convert to rgb
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.array(image)

    # Load the image to the device
    from numba import cuda

    d_image = cuda.to_device(image)

    # Create 2d array for the grayscale image
    buffer_1 = cuda.device_array(image.shape[:2], dtype=np.float32)

    # Perform the grayscale operation
    threads_per_block, blocks_per_grid = get_grid_block(image)
    grayscale_kernel[blocks_per_grid, threads_per_block](d_image, buffer_1)
    cuda.synchronize()

    if args.bw:
        # save the grayscale image
        save_image(buffer_1, args.outputImage)
        return 0

    buffer_2 = cuda.device_array(image.shape[:2], dtype=np.float32)
    # apply the gaussian filter
    gaussian_local_copy_kernel[blocks_per_grid, threads_per_block](buffer_1, buffer_2)
    cuda.synchronize()

    if args.gauss:
        # save the gaussian image
        save_image(buffer_2, args.outputImage)
        return 0

    # apply the sobel filter
    sobel_kernel[blocks_per_grid, threads_per_block](buffer_2, buffer_1)
    cuda.synchronize()

    if args.sobel:
        # save the sobel image
        save_image(buffer_1, args.outputImage)
        return 0

    # apply the threshold
    threshold_kernel[blocks_per_grid, threads_per_block](buffer_1, buffer_2, LOW_THRESHOLD,
                                                         HIGH_THRESHOLD)
    cuda.synchronize()

    if args.threshold:
        # save the threshold image
        save_image(buffer_2, args.outputImage)
        return 0

    # apply hysteresis
    hysterisis_kernel[blocks_per_grid, threads_per_block](buffer_2, buffer_1)
    cuda.synchronize()

    if args.hysteresis:
        # save the hysteresis image
        save_image(buffer_1, args.outputImage)
        return 0


def save_image(d_image, output_image_path):
    output_image = Image.fromarray(d_image.copy_to_host().astype(np.uint8))
    output_image.save(output_image_path)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tb', type=int, help='size of a thread block for all operations')
    parser.add_argument('--bw', action='store_true', help='perform only the bw_kernel')
    parser.add_argument('--gauss', action='store_true', help='perform the bw_kernel and the gauss_kernel')
    parser.add_argument('--sobel', action='store_true',
                        help='perform all kernels up to sobel_kernel  and write to disk the magnitude of each pixel')
    parser.add_argument('--non_max_suppressed', action='store_true', help='perform the non_max_suppressed')
    parser.add_argument('--threshold', action='store_true', help='perform all kernels up to threshold_kernel')
    parser.add_argument('--hysteresis', action='store_true', help='perform hysteresis')
    parser.add_argument('inputImage', type=str, help='the source image')
    parser.add_argument('outputImage', type=str, help='the destination image')

    args = parser.parse_args()

    if args.tb:
        print(f"Using thread block of size {args.tb}")
        DEFAULT_THREADS_PER_BLOCK = args.tb, args.tb

    main(args)