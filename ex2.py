import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from common import calculate_num_frames

import logging

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Function to read YUV420 frames and return only Y-plane
def read_y_component(file_path, width, height, num_frames):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)  # For U and V planes, which will be skipped

    with open(file_path, 'rb') as file:
        for _ in range(num_frames):
            y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

            # Skip U and V planes
            file.read(uv_size)
            file.read(uv_size)

            yield y_plane


# Function to save Y-only frames to individual files
def save_y_frames(input_file, output_file, width, height):
    num_frames = calculate_num_frames(input_file, width, height)
    if os.path.exists(output_file):
        logger.info(f"y only {output_file} already exists. skipping...")
        return
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        for frame_index, y_plane in enumerate(read_y_component(input_file, width, height, num_frames)):
            f_out.write(y_plane.tobytes())


def pad_frame(frame, block_size, pad_value=128):
    height, width = frame.shape
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        padded_frame = np.full((height + pad_height, width + pad_width), pad_value, dtype=np.uint8)
        padded_frame[:height, :width] = frame
        return padded_frame
    return frame


# Function to split the frame into blocks of size (block_size x block_size)
def split_into_blocks(frame, block_size):
    height, width = frame.shape
    return (frame.reshape(height // block_size, block_size, -1, block_size)
                 .swapaxes(1, 2)
                 .reshape(-1, block_size, block_size))


def calculate_block_average(block):
    return np.round(np.mean(block)).astype(np.uint8)

def replace_with_average(blocks):
    return np.array([np.full_like(block, calculate_block_average(block)) for block in blocks])

def reconstruct_frame_from_blocks(blocks, frame_shape, block_size):
    height, width = frame_shape
    rows = height // block_size
    cols = width // block_size
    return (blocks.reshape(rows, cols, block_size, block_size)
                   .swapaxes(1, 2)
                   .reshape(height, width))


# Function to process Y-only files and split them into blocks
def process_y_frames(input_file, width, height, block_sizes):
    logger.info(f"Processing file: {input_file}")
    file_prefix = os.path.splitext(input_file)[0]

    file_handles = {}

    # Open output files dynamically based on block sizes
    for block_size in block_sizes:
        file_name = f'{file_prefix}-{block_size}block.y'
        file_handles[block_size] = open(file_name, 'wb')

    y_size = width * height

    with open(input_file, 'rb') as f_in:
        frame_index = 0
        while True:
            y_frame = f_in.read(y_size)
            if not y_frame:
                break  # End of file

            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))

            for block_size in block_sizes:
                # Pad the frame if necessary
                padded_frame = pad_frame(y_plane, block_size)

                # Split the frame into (block_size x block_size) blocks
                blocks = split_into_blocks(padded_frame, block_size)

                # Replace each block with its average
                averaged_blocks = replace_with_average(blocks)

                # Reconstruct the frame from blocks
                reconstructed_frame = reconstruct_frame_from_blocks(averaged_blocks, padded_frame.shape, block_size)

                # Write the reconstructed frame to the respective output file
                file_handles[block_size].write(reconstructed_frame.tobytes())

                # Example logging to show how many blocks are generated
                logger.info(f"Frame {frame_index}: Processed {len(blocks)} blocks of size {block_size}x{block_size}")

            frame_index += 1

    # Close all the file handles
    for handle in file_handles.values():
        handle.close()

def calculate_psnr_ssim(original_file, averaged_file, width, height):
    psnr_values = []
    ssim_values = []

    with open(original_file, 'rb') as orig, open(averaged_file, 'rb') as avg:
        frame_index = 0
        while True:
            orig_frame = orig.read(width * height)
            avg_frame = avg.read(width * height)
            if not orig_frame or not avg_frame:
                break  # End of file

            orig_y_plane = np.frombuffer(orig_frame, dtype=np.uint8).reshape((height, width))
            avg_y_plane = np.frombuffer(avg_frame, dtype=np.uint8).reshape((height, width))

            # Calculate PSNR and SSIM
            current_psnr = psnr(orig_y_plane, avg_y_plane)
            current_ssim = ssim(orig_y_plane, avg_y_plane)

            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)

            logger.info(f"Frame {frame_index}: PSNR = {current_psnr}, SSIM = {current_ssim}")
            frame_index += 1

    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)

    logger.info(f"Average PSNR: {average_psnr}, Average SSIM: {average_ssim}")
    return average_psnr, average_ssim

# g. Plot graphs
def plot_quality_metrics(block_sizes, psnr_values, ssim_values):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(block_sizes, psnr_values, marker='o', label='PSNR')
    plt.title('PSNR vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('PSNR (dB)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(block_sizes, ssim_values, marker='o', label='SSIM', color='orange')
    plt.title('SSIM vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('SSIM')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(input_file, width, height):
    file_prefix = os.path.splitext(input_file)[0]
    y_only_file = f'{file_prefix}.y'
    save_y_frames(input_file, y_only_file, width, height)

    block_sizes = [2, 8, 16, 64]  # Block sizes to process
    process_y_frames(y_only_file, width, height, block_sizes)

    for block_size in block_sizes:
        averaged_file = f'{file_prefix}-{block_size}block.y'
        average_psnr, average_ssim = calculate_psnr_ssim(y_only_file, averaged_file, width, height)
        # Store results for plotting
        if 'psnr_results' not in locals():
            psnr_results = []
            ssim_results = []
        psnr_results.append(average_psnr)
        ssim_results.append(average_ssim)

    plot_quality_metrics(block_sizes, psnr_results, ssim_results)


