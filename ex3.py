import logging
import os
import numpy as np

from common import mae, pad_frame, split_into_blocks
from ex2 import read_y_component

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_best_match(curr_block, prev_frame, block_size, x, y, search_range):
    """Find the best matching block within a given search range."""
    height, width = prev_frame.shape
    min_mae = float('inf')
    best_mv = (0, 0)  # motion vector (dx, dy)

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_x = x + dx
            ref_y = y + dy

            # Check if the reference block is within bounds
            if ref_x < 0 or ref_y < 0 or ref_x + block_size > width or ref_y + block_size > height:
                continue

            ref_block = prev_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            # Choose the block with the smallest MAE, breaking ties as described
            if error < min_mae or (error == min_mae and abs(dx) + abs(dy) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = (dx, dy)

    return best_mv, min_mae

def motion_estimation(curr_frame, prev_frame, block_size, search_range):
    height, width = curr_frame.shape
    mv_field = []  # store motion vectors for each block
    avg_mae = 0

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = curr_frame[y:y + block_size, x:x + block_size]
            best_mv, mae_value = find_best_match(curr_block, prev_frame, block_size, x, y, search_range)
            mv_field.append((x, y, best_mv[0], best_mv[1]))  # store motion vector (block coordinates + motion vector)
            avg_mae += mae_value

    avg_mae /= (height // block_size) * (width // block_size)
    return mv_field, avg_mae


def main(input_file, width, height):
    file_prefix = os.path.splitext(input_file)[0]
    input_file = f'{file_prefix}.y'

    # search_ranges = [1, 4, 8]  # Search ranges to test
    # block_sizes = [2, 8, 16, 64]  # Block sizes to process
    search_ranges = [ 1]  # Search ranges to test
    block_sizes = [16]  # Block sizes to process

    # Number of frames to process (for example, first 12 frames)
    num_frames = 12

    # Assuming a previous frame of all 128 (as specified for the first frame)
    prev_frame = np.full((height, width), 128, dtype=np.uint8)

    y_size = width * height

    # Open the input file containing Y frames
    with open(input_file, 'rb') as f_in:
        frame_index = 0
        while True:
            y_frame = f_in.read(y_size)
            if not y_frame:
                break  # End of file
            logger.debug(f"Processing frame {frame_index + 1}/{num_frames}")
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))

            for block_size in block_sizes:
                # Pad the frame if necessary
                padded_frame = pad_frame(y_plane, block_size)

                for search_range in search_ranges:
                    logger.info(
                        f"Frame {frame_index + 1}, Block Size {block_size}x{block_size}, Search Range {search_range}")

                    # Perform motion estimation
                    mv_field, avg_mae = motion_estimation(padded_frame, prev_frame, block_size, search_range)

                    # Save the motion vectors and MAE for the current frame, block size, and search range
                    mv_output_file = f'{file_prefix}_frame{frame_index + 1}_block{block_size}_search{search_range}_mv.txt'
                    with open(mv_output_file, 'w') as mv_out:
                        for mv in mv_field:
                            mv_out.write(f'{mv[0]} {mv[1]} {mv[2]} {mv[3]}\n')  # (x, y, dx, dy)
                        mv_out.write(f'Average MAE: {avg_mae}\n')

                    logger.info(f"Average MAE for Block Size {block_size} and Search Range {search_range}: {avg_mae}")

            # Set the current frame as the previous frame for the next iteration
            prev_frame = y_plane

            # Stop after processing the first 12 frames
            if frame_index + 1 >= num_frames:
                break
