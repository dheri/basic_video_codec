import concurrent
import concurrent.futures
import csv
import time
from contextlib import ExitStack

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from block_predictor import predict_block
from common import pad_frame, get_logger
from decoder import find_predicted_block
from encoder.dct import *
from encoder.params import EncoderParameters, EncodedFrame, EncodedBlock
from file_io import write_mv_to_file, write_y_only_frame, FileIOHelper
from input_parameters import InputParameters


logger = get_logger()
def encode(params: InputParameters):
    file_io = FileIOHelper(params)

    start_time = time.time()
    y_size = params.width * params.height
    prev_frame = np.full((params.height, params.width), 128, dtype=np.uint8)
    with ExitStack() as stack:
        f_in = stack.enter_context(open(params.y_only_file, 'rb'))
        mv_fh = stack.enter_context(open(file_io.get_mv_file_name(), 'wt'))
        quant_dct_coff_fh =stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'wb'))
        residual_yuv_fh = stack.enter_context(open(file_io.get_mc_residual_file_name(), 'wb'))
        reconstructed_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'wb'))

        metrics_csv_fh = stack.enter_context(open(file_io.get_metrics_csv_file_name(), 'wt', newline=''))
        frames_to_process = params.frames_to_process
        height = params.height
        width = params.width
        block_size = params.encoder_parameters.block_size
        search_range = params.encoder_parameters.search_range


        metrics_csv_writer = csv.writer(metrics_csv_fh)
        metrics_csv_writer.writerow(['Frame Index', 'Average MAE', 'PSNR'])
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

            # logger.info(f"Frame {frame_index }, Block Size {block_size}x{block_size}, Search Range {search_range}")
            # mv_field, avg_mae, residual, reconstructed_frame, residual_frame = encode_frame(padded_frame, prev_frame, block_size, search_range, residual_approx_factor)
            encoded_frame : EncodedFrame = encode_frame(padded_frame, prev_frame, params.encoder_parameters )
            mv_field = encoded_frame.mv_field
            avg_mae = encoded_frame.avg_mae
            psnr = peak_signal_noise_ratio(padded_frame, encoded_frame.reconstructed_frame_with_mc)


            logger.info(f"{frame_index:2}: i={block_size} r={search_range}, mae [{round(avg_mae,2):7.2f}] psnr [{round(psnr,2):6.2f}]")
            write_mv_to_file(mv_fh, mv_field)
            write_y_only_frame(reconstructed_fh, encoded_frame.reconstructed_frame_with_mc)
            write_y_only_frame(residual_yuv_fh, encoded_frame.residual_frame_with_mc)
            write_y_only_frame(quant_dct_coff_fh, encoded_frame.quat_dct_coffs_with_mc)


            metrics_csv_writer.writerow([frame_index, avg_mae, psnr])
            prev_frame = encoded_frame.reconstructed_frame_with_mc


    end_time = time.time()
    elapsed_time = end_time - start_time

    num_of_blocks = (height // block_size) * (width // block_size)
    num_of_comparisons = num_of_blocks * (2 * search_range + 1) ** 2
    result = str(f"{num_of_comparisons/elapsed_time:9.3f} | {num_of_comparisons:7d} | {num_of_blocks/elapsed_time:7.3f} |  {num_of_blocks:5d} | {frames_to_process/elapsed_time:6.2f} | {frames_to_process:3d} | {elapsed_time:6.3f} | {block_size:2d} | {search_range:2d} |\n")
    print(result)
    with open('../results.csv', 'at') as f_in:
        f_in.write(result)
    print('end encoding')
    return


def encode_frame(curr_frame, prev_frame, encoder_params: EncoderParameters):
    block_size = encoder_params.block_size
    search_range = encoder_params.search_range
    quantization_factor = encoder_params.quantization_factor

    height, width = curr_frame.shape
    num_of_blocks = (height // block_size) * (width // block_size)
    mv_field = {}
    mae_of_blocks = 0

    # Function to process each block
    def process_block(y, x):
        curr_block = curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)

        # Adjust search range to avoid going out of frame boundaries
        prev_partial_frame_x_start_idx = max(x - search_range, 0)
        prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)
        prev_partial_frame_y_start_idx = max(y - search_range, 0)
        prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)

        prev_partial_frame = prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                             prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

        # Predict the block using motion estimation
        best_mv_within_search_window, best_match_mae, best_match_block = predict_block(curr_block, prev_partial_frame, block_size)

        # Calculate the motion vector relative to the full frame
        motion_vector = [best_mv_within_search_window[0] + prev_partial_frame_x_start_idx - x,
                         best_mv_within_search_window[1] + prev_partial_frame_y_start_idx - y]

        mv_field[(x, y)] = motion_vector

        # Apply the motion vector to find the predicted block
        predicted_block_with_mc = find_predicted_block(motion_vector, x, y, prev_frame, block_size).astype(np.int16)

        # Compute the residual block using motion compensation
        residual_block_with_mc = np.subtract(curr_block, predicted_block_with_mc)

        # Apply 2D DCT to the residual block
        dct_coffs = apply_dct_2d(residual_block_with_mc)

        # Clip the DCT coefficients to avoid overflow (range: -128 to 127)
        # clipped_dct_coffs = np.clip(dct_coffs, -128, 127).astype(np.int16)

        # Apply quantization
        Q = generate_quantization_matrix(block_size, quantization_factor)
        quantized_dct_coffs = quantize_block(dct_coffs, Q)

        # logger.info(f'min/max: [ {np.min(quantized_dct_coffs)} / {np.max(quantized_dct_coffs)}]')

        quantized_dct_coffs = np.clip(quantized_dct_coffs, -128, 127).astype(np.int8)
        # Rescale and apply inverse DCT
        rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
        idct_residual_block = apply_idct_2d(rescaled_dct_coffs)

        # Reconstruct the block
        reconstructed_block_with_mc = np.round(idct_residual_block + predicted_block_with_mc).astype(np.int16)

        # Clip the reconstructed block to the valid range for uint8 (0 to 255)
        clipped_reconstructed_block = np.clip(reconstructed_block_with_mc, 0, 255).astype(np.uint8)

        return EncodedBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block,
                            clipped_reconstructed_block )

    # Process all blocks in the frame
    reconstructed_frame_with_mc = np.zeros_like(curr_frame, dtype=np.uint8)
    residual_frame_with_mc = np.zeros_like(curr_frame, dtype=np.int8)
    quat_dct_coffs_frame_with_mc = np.zeros_like(curr_frame, dtype=np.int8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_block, y, x) for y in range(0, height, block_size) for x in range(0, width, block_size)]
        for future in concurrent.futures.as_completed(futures):
            encoded_block = future.result()
            block_cords = encoded_block.block_coords
            x, y = block_cords

            reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block_with_mc
            residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
            quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

            mv_field[block_cords] = encoded_block.motion_vector
            mae_of_blocks += encoded_block.mae

    avg_mae = mae_of_blocks / num_of_blocks
    return EncodedFrame(mv_field, avg_mae, residual_frame_with_mc, quat_dct_coffs_frame_with_mc, reconstructed_frame_with_mc)

def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple
