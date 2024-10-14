import concurrent
import logging
import time

import numpy as np
import concurrent.futures

from block_predictor import predict_block
from common import pad_frame
from file_io import write_to_file, write_y_only_frame

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def encode(input_file, mv_output_file, residual_yuv_file, reconstructed_file, frames_to_process, height, width, block_size, search_range, residual_approx_factor):
    start_time = time.time()
    y_size = width * height
    prev_frame = np.full((height, width), 128, dtype=np.uint8)
    avg_mae_per_frame = list()
    with open(input_file, 'rb') as f_in, open(mv_output_file, 'wt') as mv_fh, open(residual_yuv_file, 'wb') as residual_yuv_fh, open(reconstructed_file, 'wb') as reconstructed_fh:
        frame_index = 0
        while True:
            frame_index += 1
            y_frame = f_in.read(y_size)
            if not y_frame or frame_index > frames_to_process:
                break  # End of file or end of frames
            logger.debug(f"Processing frame {frame_index}/{frames_to_process}")
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))
            padded_frame = pad_frame(y_plane, block_size)
            # padded_frame = y_plane

            logger.info(f"Frame {frame_index }, Block Size {block_size}x{block_size}, Search Range {search_range}")
            mv_field, avg_mae, residual, reconstructed_frame, residual_frame = encode_frame(padded_frame, prev_frame, block_size, search_range, residual_approx_factor)

            write_to_file(mv_fh, frame_index, mv_field)
            # write_to_file(residual_txt_fh, frame_index, residual, True)
            write_y_only_frame(reconstructed_fh, reconstructed_frame)
            write_y_only_frame(residual_yuv_fh, residual_frame)

            logger.info(f"Average MAE for Block Size {block_size} and Search Range {search_range}: {avg_mae}")
            avg_mae_per_frame.append(avg_mae)
            prev_frame = reconstructed_frame


    end_time = time.time()
    elapsed_time = end_time - start_time

    num_of_blocks = (height // block_size) * (width // block_size)
    num_of_comparisons = num_of_blocks * (2 * search_range + 1) ** 2
    result = str(f"{num_of_comparisons/elapsed_time:9.3f} | {num_of_comparisons:7d} | {num_of_blocks/elapsed_time:7.3f} |  {num_of_blocks:5d} | {frames_to_process/elapsed_time:6.2f} | {frames_to_process:3d} | {elapsed_time:6.3f} | {block_size:2d} | {search_range:2d} |\n")
    print(result)
    with open('results.csv', 'at') as f_in:
        f_in.write(result)
    print('end encoding')
    return  avg_mae_per_frame


def encode_frame(curr_frame, prev_frame, block_size, search_range, residual_approx_factor):
    if curr_frame.shape != prev_frame.shape:
        raise ValueError("Motion estimation got mismatch in frame shapes")

    height, width = curr_frame.shape
    num_of_blocks = (height // block_size) * (width // block_size)
    mv_field = {}
    # residuals = {}
    mae_of_blocks = 0

    # Function to process each block (for threading)
    def process_block(y, x):
        curr_block = curr_frame[y:y + block_size, x:x + block_size]

        # Adjust search range to avoid going out of frame boundaries
        prev_partial_frame_x_start_idx = max(x - search_range, 0)
        prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)
        prev_partial_frame_y_start_idx = max(y - search_range, 0)
        prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)

        prev_partial_frame = prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                             prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

        best_mv_within_search_window, best_match_mae, best_match_block  = predict_block(curr_block, prev_partial_frame, block_size)

        # Ensure motion vector is calculated relative to the entire frame
        motion_vector = [best_mv_within_search_window[0] + prev_partial_frame_x_start_idx - x,
                         best_mv_within_search_window[1] + prev_partial_frame_y_start_idx - y]

        mv_field[(x, y)] = motion_vector
        # Generate the predicted block by shifting the previous frame based on the motion vector
        predicted_block_with_mc = prev_frame[y + motion_vector[1]:y + motion_vector[1] + block_size,
                                             x + motion_vector[0]:x + motion_vector[0] + block_size]

        # Residuals with motion compensation
        residual_block_with_mc = np.subtract(curr_block, predicted_block_with_mc)

        # Residuals without motion compensation (using the same position in previous frame)
        prev_block_no_mc = prev_frame[y:y + block_size, x:x + block_size]
        residual_block_without_mc = np.subtract(curr_block, prev_block_no_mc)

        # Optionally apply residual approximation to both
        approx_residual_with_mc = round_to_nearest_multiple(residual_block_with_mc, residual_approx_factor)
        approx_residual_without_mc = round_to_nearest_multiple(residual_block_without_mc, residual_approx_factor)

        # Reconstruct the block using both methods
        reconstructed_block_with_mc = approx_residual_with_mc + predicted_block_with_mc
        reconstructed_block_without_mc = approx_residual_without_mc + prev_block_no_mc

        return {
            'block_coords': (x, y),
            'motion_vector': motion_vector,
            'mae': best_match_mae,
            'residual_with_mc': approx_residual_with_mc,
            'reconstructed_with_mc': reconstructed_block_with_mc,
            'residual_without_mc': approx_residual_without_mc,
            'reconstructed_without_mc': reconstructed_block_without_mc,
        }

    reconstructed_frame_with_mc = np.zeros_like(curr_frame)
    reconstructed_frame_without_mc = np.zeros_like(curr_frame)
    residual_frame_with_mc = np.zeros_like(curr_frame)
    residual_frame_without_mc = np.zeros_like(curr_frame)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                futures.append(executor.submit(process_block, y, x))

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            block_data = future.result()
            block_cords  = block_data['block_coords']
            x = block_cords[0]
            y = block_cords[1]
            mv =  block_data['motion_vector']


            # Update reconstructed and residual frames
            reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = block_data['reconstructed_with_mc']
            reconstructed_frame_without_mc[y:y + block_size, x:x + block_size] = block_data['reconstructed_without_mc']

            residual_frame_with_mc[y:y + block_size, x:x + block_size] = block_data['residual_with_mc']
            residual_frame_without_mc[y:y + block_size, x:x + block_size] = block_data['residual_without_mc']


            mv_field[block_cords] = mv
            # residuals[block_cords] = mc_residual_b
            mae_of_blocks += block_data['mae']

    avg_mae = mae_of_blocks /num_of_blocks
    # return mv_field, avg_mae, residuals, reconstructed_frame, residual_frame
    return {
        'mv_field': mv_field,
        'avg_mae': avg_mae,
        'residual_frame_with_mc': residual_frame_with_mc,
        'residual_frame_without_mc': residual_frame_without_mc,
        'reconstructed_frame_with_mc': reconstructed_frame_with_mc,
        'reconstructed_frame_without_mc': reconstructed_frame_without_mc,
    }


def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple
