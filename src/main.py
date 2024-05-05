from pathlib import Path

import utils
import numpy as np
from PIL import Image

from hysteresis import hysterisis_kernel
from non_max_suppression import non_max_suppression_kernel
from sobel import sobel_kernel
from src.grayscale import grayscale_kernel
from src.gaussian import gaussian_kernel, gaussian_local_copy_kernel
from threshold import threshold_kernel

LOW_THRESHOLD = 51
HIGH_THRESHOLD = 102


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
    d_output_grayscale = cuda.device_array(image.shape[:2], dtype=np.float32)

    # Perform the grayscale operation
    threads_per_block, blocks_per_grid = utils.get_grid_block(image)
    grayscale_kernel[blocks_per_grid, threads_per_block](d_image, d_output_grayscale)
    cuda.synchronize()

    if args.bw:
        # save the grayscale image
        save_image(d_output_grayscale, args.outputImage)
        return 0

    d_output_gaussian = cuda.device_array(image.shape[:2], dtype=np.float32)
    # apply the gaussian filter
    gaussian_local_copy_kernel[blocks_per_grid, threads_per_block](d_output_grayscale, d_output_gaussian)
    cuda.synchronize()

    if args.gauss:
        # save the gaussian image
        save_image(d_output_gaussian, args.outputImage)
        return 0

    # apply the sobel filter
    sobel_kernel[blocks_per_grid, threads_per_block](d_output_gaussian, d_output_grayscale)
    cuda.synchronize()

    if args.sobel:
        # save the sobel image
        save_image(d_output_grayscale, args.outputImage)
        return 0

    cuda.synchronize()

    d_output_threshold = cuda.device_array(image.shape[:2], dtype=np.float32)
    # apply the threshold
    threshold_kernel[blocks_per_grid, threads_per_block](d_output_grayscale, d_output_threshold, LOW_THRESHOLD,
                                                         HIGH_THRESHOLD)
    cuda.synchronize()

    if args.threshold:
        # save the threshold image
        save_image(d_output_threshold, args.outputImage)
        return 0

    # apply hysteresis
    d_output_hysteresis = cuda.device_array(image.shape[:2], dtype=np.uint8)
    hysterisis_kernel[blocks_per_grid, threads_per_block](d_output_threshold, d_output_hysteresis)
    cuda.synchronize()

    if args.hysteresis:
        # save the hysteresis image
        save_image(d_output_hysteresis, args.outputImage)
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
    parser.add_argument('--threshold', action='store_true', help='perform all kernels up to threshold_kernel')
    parser.add_argument('inputImage', type=str, help='the source image')
    parser.add_argument('outputImage', type=str, help='the destination image')

    args = parser.parse_args()

    if args.tb:
        print(f"Using thread block of size {args.tb}")
        utils.DEFAULT_THREADS_PER_BLOCK = args.tb, args.tb

    main(args)
