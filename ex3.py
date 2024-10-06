import concurrent
import logging
import os
import time
import cupy as cp

import numpy as np
from matplotlib import pyplot as plt

from common import mae, pad_frame

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def find_lowest_mae_block(curr_block, prev_partial_frame, block_size):
    """Find the block with the lowest MAE from a smaller previous partial frame."""
    height, width = prev_partial_frame.shape
    min_mae = float('inf')
    best_mv = (0, 0)  # motion vector (dx, dy)

    # Loop through all possible positions in the previous partial frame
    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = (ref_x, ref_y)

    return best_mv, min_mae

def find_lowest_mae_block_gpu(curr_block, prev_partial_frame, block_size):
    """Find the block with the lowest MAE from a smaller previous partial frame, using GPU."""
    height, width = prev_partial_frame.shape
    min_mae = float('inf')
    best_mv = (0, 0)  # motion vector (dx, dy)

    # Prepare for vectorized operations
    curr_block_gpu = cp.asarray(curr_block)

    # Loop through all possible positions in the previous partial frame on the GPU
    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block_gpu = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

            # Calculate MAE using CuPy's vectorized operations
            error_gpu = cp.mean(cp.abs(curr_block_gpu - ref_block_gpu))

            # Update best match if a lower MAE is found
            if error_gpu < min_mae:
                min_mae = error_gpu
                best_mv = (ref_x, ref_y)

    return best_mv, min_mae

def get_prev_partial_frame(prev_frame_gpu, y, x, block_size, search_range, height, width):
    prev_partial_frame_y_start_idx = max(y - search_range, 0)
    prev_partial_frame_x_start_idx = max(x - search_range, 0)
    prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)
    prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)

    return prev_frame_gpu[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                           prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]


def motion_estimation(curr_frame, prev_frame, block_size, search_range):
    if curr_frame.shape != prev_frame.shape:
        raise ValueError("Motion estimation got mismatch in frame shapes")

    height, width = curr_frame.shape
    num_of_blocks = (height // block_size) * (width // block_size)
    mv_field = []
    avg_mae = 0

    # Convert frames to CuPy arrays to minimize data transfers
    curr_frame_gpu = cp.asarray(curr_frame)
    prev_frame_gpu = cp.asarray(prev_frame)

    # Process each block on the GPU
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = curr_frame_gpu[y:y + block_size, x:x + block_size]
            prev_partial_frame = get_prev_partial_frame(prev_frame_gpu, y, x, block_size, search_range, height, width)

            best_mv, mae_value = find_lowest_mae_block_gpu(curr_block, prev_partial_frame, block_size)
            mv_field.append((x, y, best_mv[0], best_mv[1]))
            avg_mae += mae_value

    avg_mae /= num_of_blocks
    return mv_field, avg_mae


def plot_metrics(insights):
    avg_mae_values = [insight['avg_mae'] for insight in insights]  # Extract avg_mae for each frame
    frame_numbers = range(1, len(avg_mae_values) + 1)  # Generate frame numbers

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')

    plt.title('Average MAE per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Average MAE')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()
    plt.savefig('mae_per_frame.png')
    # plt.imsave()

def main(input_file, width, height):
    start_time = time.time()

    file_prefix = os.path.splitext(input_file)[0]
    input_file = f'{file_prefix}.y'

    # search_ranges = [1, 4, 8]  # Search ranges to test
    # block_sizes = [2, 8, 16, 64]  # Block sizes to process
    search_ranges = [ 2]  # Search ranges to test 'r'
    block_sizes = [8]  # Block sizes to process 'i'

    frames_to_process = 22

    # Assuming a previous frame of all 128 (as specified for the first frame)
    prev_frame = np.full((height, width), 128, dtype=np.uint8)

    y_size = width * height
    insights = list()

    # Open the input file containing Y frames
    with open(input_file, 'rb') as f_in:
        frame_index = 0
        while True:
            frame_index += 1
            insights_dict = dict()
            y_frame = f_in.read(y_size)
            if not y_frame or frame_index > frames_to_process:
                break  # End of file or end of frames
            logger.debug(f"Processing frame {frame_index}/{frames_to_process}")
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))

            for block_size in block_sizes:
                # Pad the frame if necessary
                padded_frame = pad_frame(y_plane, block_size)

                for search_range in search_ranges:
                    logger.info(f"Frame {frame_index }, Block Size {block_size}x{block_size}, Search Range {search_range}")

                    insights_dict['mv_field'], insights_dict['avg_mae'] = motion_estimation(padded_frame, prev_frame, block_size, search_range)

                    # Save the motion vectors and MAE for the current frame, block size, and search range
                    # # mv_output_file = f'{file_prefix}_frame{frame_index}_block{block_size}_search{search_range}_mv.txt'
                    # with open(mv_output_file, 'w') as mv_out:
                    #     for mv in mv_field:
                    #         mv_out.write(f'{mv[0]} {mv[1]} {mv[2]} {mv[3]}\n')  # (x, y, dx, dy)
                    #     mv_out.write(f'Average MAE: {avg_mae}\n')

                    logger.info(f"Average MAE for Block Size {block_size} and Search Range {search_range}: {insights_dict['avg_mae']}")

            # Set the current frame as the previous frame for the next iteration
            prev_frame = y_plane
            insights.append(insights_dict)

    plot_metrics(insights)
    end_time = time.time()
    elapsed_time = end_time - start_time

    result = str(f"{frames_to_process/elapsed_time:.2f} | {elapsed_time:.3f} | {frames_to_process} | {block_size} | {search_range}\n")
    print(result)
    with open('results.csv', 'at') as f_in:
        f_in.write(result)
    print('end')
