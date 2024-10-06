import concurrent
import logging
import os
import time
from io import BufferedWriter

import numpy as np
from matplotlib import pyplot as plt

from common import mae, pad_frame

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
def find_lowest_mae_block(curr_block, prev_partial_frame, block_size, n=1):
    """Find the block with the lowest MAE from a smaller previous partial frame."""
    height, width = prev_partial_frame.shape
    min_mae = float('inf')
    best_mv = [0,0]  # motion vector wrt origin of prev_partial_frame

    # Loop through all possible positions in the previous partial frame

    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            # Update best match if a lower MAE is found, breaking ties as described
            if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = [ref_x , ref_y]  # (dx, dy)


    residual = np.subtract(curr_block, ref_block)
    approx_residual = np.round(residual / (2**n)) * (2**n)

    return best_mv, min_mae, approx_residual

def motion_estimation(curr_frame, prev_frame, block_size, search_range, residual_approx_factor):
    if curr_frame.shape != prev_frame.shape:
        raise ValueError("Motion estimation got mismatch in frame shapes")
    height, width = curr_frame.shape
    num_of_blocks = (height // block_size) * (width // block_size)
    mv_field = {}
    residuals = {}
    avg_mae = 0

    # Function to process each block (for threading)
    def process_block(y, x):
        curr_block = curr_frame[y:y + block_size, x:x + block_size]
        reconstructed_frame = np.zeros_like(curr_frame)

        prev_partial_frame_y_start_idx = max(y - search_range, 0)
        prev_partial_frame_x_start_idx = max(x - search_range, 0)
        prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)
        prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)

        prev_partial_frame = prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                                        prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

        best_mv, mae_value, residual = find_lowest_mae_block(curr_block, prev_partial_frame, block_size, residual_approx_factor)

        # Ensure that the motion vector cannot be negative for boundary blocks
        mv_x = 0 if x - search_range < 0 else best_mv[0]
        mv_y = 0 if y - search_range < 0 else best_mv[1]

        best_mv = [mv_x, mv_y]


        reconstructed_block = reconstruct_block(curr_block, x,y, best_mv, prev_frame, block_size, residual_approx_factor)

        # Save the reconstructed block in the corresponding position in the reconstructed frame
        reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block

        # # Adjust the motion vector based on the search window's starting position
        # mv_x = best_mv[0] + prev_partial_frame_x_start_idx - x
        # mv_y = best_mv[1] + prev_partial_frame_y_start_idx - y


        return (x, y), (mv_x, mv_y), mae_value, residual, reconstructed_frame

    # Use ThreadPoolExecutor to parallelize the processing of blocks
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                futures.append(executor.submit(process_block, y, x))

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            block_cords, mv, mae_value, residual, reconstructed_frame = future.result()
            mv_field[block_cords] = mv
            residuals[block_cords] = residual
            avg_mae += mae_value

    avg_mae /= num_of_blocks
    return mv_field, avg_mae, residuals, reconstructed_frame


def plot_metrics(insights, file_prefix, block_size, search_range):
    avg_mae_values = [insight['avg_mae'] for insight in insights]  # Extract avg_mae for each frame
    frame_numbers = range(1, len(avg_mae_values) + 1)  # Generate frame numbers

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')

    plt.title(f'MAE per Frame, i = {block_size}, r = {search_range}')
    plt.xlabel('Frame Number')
    plt.ylabel('Average MAE')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_mae.png')

def write_to_file(file_handle : BufferedWriter , frame_idx , data, new_line_per_block = False):
    file_handle.write(f'\nFrame: {frame_idx}\n')
    new_line_char = f'\n' if new_line_per_block else ''
    for k in sorted(data.keys()):
        file_handle.write(f'{new_line_char}{k}:{data[k]}|')

def process_frame(curr_frame, prev_frame, block_size, search_range, n):
    """Process the frame block by block, reconstruct each block, and generate the reconstructed frame."""
    height, width = curr_frame.shape
    reconstructed_frame = np.zeros_like(curr_frame)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = curr_frame[y:y + block_size, x:x + block_size]

            # Obtain the best motion vector and mae using your existing `find_lowest_mae_block` method
            best_mv, _, _ = find_lowest_mae_block(curr_block, prev_frame, block_size)

            # Reconstruct the block by adding approximated residuals to the predictor block
            reconstructed_block = reconstruct_block(curr_block, best_mv, prev_frame, block_size, n)

            # Save the reconstructed block in the corresponding position in the reconstructed frame
            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block

    return reconstructed_frame
def reconstruct_block(curr_block, x, y, best_mv, prev_frame, block_size, n):
    """Reconstruct the block by adding the approximated residual to the predicted block."""
    ref_x = best_mv[0] +  x
    ref_y = best_mv[1] +  y

    # Predictor block from the previous frame based on the motion vector
    predictor_block = prev_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

    # Residual block
    residual_block = curr_block - predictor_block

    # Approximate residual block (rounding to the nearest multiple of 2^n)
    approximated_residual = round_to_nearest_multiple(residual_block, n)

    # Reconstructed block: predictor block + approximated residual
    reconstructed_block = predictor_block + approximated_residual
    return reconstructed_block

def round_to_nearest_multiple(arr, n):
    """Round every element in arr to the nearest multiple of 2^n."""
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple

def write_y_only_reconstructed_frame(file_handle, reconstructed_frame):
    """Write the Y-only reconstructed frame to a file."""
    file_handle.write(reconstructed_frame.tobytes())  # Write the raw Y frame to file


def main(input_file, width, height):
    start_time = time.time()

    file_prefix = os.path.splitext(input_file)[0]
    input_file = f'{file_prefix}.y'

    search_ranges = [1, 2, 4, 8]  # Search ranges to test
    block_sizes = [2, 8, 16, 64]  # Block sizes to process
    search_range = search_ranges[1]
    block_size = block_sizes[1]  # Block sizes to process 'i'
    residual_approx_factor = 4
    frames_to_process = 3

    # Assuming a previous frame of all 128 (as specified for the first frame)
    prev_frame = np.full((height, width), 128, dtype=np.uint8)

    y_size = width * height
    insights = list()

    file_identifier = f'{block_size}_{search_range}_{residual_approx_factor}'

    mv_output_file = f'{file_prefix}_{file_identifier}_mv.txt'
    residual_output_file = f'{file_prefix}_{file_identifier}_residual.txt'
    reconstructed_file = f'{file_prefix}_{file_identifier}_recons.yuv'

    # Open the input file containing Y frames
    with open(input_file, 'rb') as f_in, open(mv_output_file, 'wt') as mv_fh, open(residual_output_file, 'wt') as residual_fh, open(reconstructed_file, 'wb') as reconstructed_fh:
        frame_index = 0
        while True:
            frame_index += 1
            insights_dict = dict()
            y_frame = f_in.read(y_size)
            if not y_frame or frame_index > frames_to_process:
                break  # End of file or end of frames
            logger.debug(f"Processing frame {frame_index}/{frames_to_process}")
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))
            padded_frame = pad_frame(y_plane, block_size)

            logger.info(f"Frame {frame_index }, Block Size {block_size}x{block_size}, Search Range {search_range}")
            # mv_field, avg_mae, residual = motion_estimation(padded_frame, prev_frame, block_size, search_range,residual_approx_factor)
            mv_field, avg_mae, residual, reconstructed_frame = motion_estimation(padded_frame, prev_frame, block_size, search_range,residual_approx_factor)

            write_to_file(mv_fh, frame_index, mv_field)
            write_to_file(residual_fh, frame_index, residual, True)
            write_y_only_reconstructed_frame(reconstructed_fh, reconstructed_frame)


            insights_dict['mv_field'], insights_dict['avg_mae'], insights_dict['residual'] = mv_field, avg_mae, residual
            logger.info(f"Average MAE for Block Size {block_size} and Search Range {search_range}: {insights_dict['avg_mae']}")
            insights.append(insights_dict)
            # Set the current frame as the previous frame for the next iteration
            prev_frame = reconstructed_frame


    end_time = time.time()
    elapsed_time = end_time - start_time
    plot_metrics(insights, f'{file_prefix}_{file_identifier}', block_size, search_range)

    num_of_blocks = (height // block_size) * (width // block_size)
    num_of_comparisons = num_of_blocks * (2 * search_range + 1) ** 2
    result = str(f"{num_of_comparisons/elapsed_time:.3f} | {num_of_comparisons} | {num_of_blocks/elapsed_time:.3f} |  {num_of_blocks} | {frames_to_process/elapsed_time:.2f} | {frames_to_process} | {elapsed_time:.3f} | {block_size} | {search_range} |\n")
    print(result)
    with open('results.csv', 'at') as f_in:
        f_in.write(result)
    print('end')


