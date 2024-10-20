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

            encoded_frame : EncodedFrame = encode_frame(padded_frame, prev_frame, params.encoder_parameters )
            mv_field = encoded_frame.mv_field
            avg_mae = encoded_frame.avg_mae
            psnr = peak_signal_noise_ratio(padded_frame, encoded_frame.reconstructed_frame_with_mc)

            dct_coffs_extremes = encoded_frame.get_quat_dct_coffs_extremes()
            logger.info(f"{frame_index:2}: i={block_size} r={search_range}, mae [{round(avg_mae,2):7.2f}] psnr [{round(psnr,2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")
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

def get_motion_vector(curr_block, prev_frame, x, y, block_size, search_range, width, height):
    prev_partial_frame_x_start_idx = max(x - search_range, 0)
    prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)
    prev_partial_frame_y_start_idx = max(y - search_range, 0)
    prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)

    prev_partial_frame = prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                                    prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

    best_mv_within_search_window, best_match_mae, best_match_block = predict_block(curr_block, prev_partial_frame, block_size)

    motion_vector = [best_mv_within_search_window[0] + prev_partial_frame_x_start_idx - x,
                     best_mv_within_search_window[1] + prev_partial_frame_y_start_idx - y]

    return motion_vector, best_match_mae

# Helper function to generate residual and predicted block
def generate_residual_block(curr_block, prev_frame, motion_vector, x, y, block_size):
    predicted_block_with_mc = find_predicted_block(motion_vector, x, y, prev_frame, block_size).astype(np.int16)
    residual_block_with_mc = np.subtract(curr_block, predicted_block_with_mc)
    return predicted_block_with_mc, residual_block_with_mc

# Helper function to apply DCT and quantization
def apply_dct_and_quantization(residual_block, block_size, quantization_factor):
    dct_coffs = apply_dct_2d(residual_block)
    Q = generate_quantization_matrix(block_size, quantization_factor)
    quantized_dct_coffs = quantize_block(dct_coffs, Q)
    return quantized_dct_coffs, Q

# Helper function to reconstruct the block
def reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc):
    rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
    idct_residual_block = apply_idct_2d(rescaled_dct_coffs)
    reconstructed_block_with_mc = np.round(idct_residual_block + predicted_block_with_mc).astype(np.int16)
    clipped_reconstructed_block = np.clip(reconstructed_block_with_mc, 0, 255).astype(np.uint8)
    return clipped_reconstructed_block, idct_residual_block

# The main process block function
def process_block(curr_frame, prev_frame, x, y, block_size, search_range, quantization_factor, width, height, mv_field):
    curr_block = curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)

    # Get motion vector and MAE
    motion_vector, best_match_mae = get_motion_vector(curr_block, prev_frame, x, y, block_size, search_range, width, height)
    mv_field[(x, y)] = motion_vector

    # Generate residual and predicted block
    predicted_block_with_mc, residual_block_with_mc = generate_residual_block(curr_block, prev_frame, motion_vector, x, y, block_size)

    # Apply DCT and quantization
    quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

    # Reconstruct the block using the predicted and inverse DCT
    clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc)

    return EncodedBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block, clipped_reconstructed_block)

def encode_frame(curr_frame, prev_frame, encoder_params: EncoderParameters):
    block_size = encoder_params.block_size
    search_range = encoder_params.search_range
    quantization_factor = encoder_params.quantization_factor

    height, width = curr_frame.shape
    num_of_blocks = (height // block_size) * (width // block_size)
    mv_field = {}
    mae_of_blocks = 0

    # Initialize output frames
    reconstructed_frame_with_mc = np.zeros_like(curr_frame, dtype=np.uint8)
    residual_frame_with_mc = np.zeros_like(curr_frame, dtype=np.int8)
    quat_dct_coffs_frame_with_mc = np.zeros_like(curr_frame, dtype=np.int16)

    # Process blocks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_block, curr_frame, prev_frame, x, y, block_size, search_range, quantization_factor, width, height, mv_field)
                   for y in range(0, height, block_size)
                   for x in range(0, width, block_size)]

        for future in concurrent.futures.as_completed(futures):
            encoded_block = future.result()
            block_cords = encoded_block.block_coords
            x, y = block_cords

            # Update frames with the encoded block data
            reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block_with_mc
            residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
            quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

            mae_of_blocks += encoded_block.mae

    avg_mae = mae_of_blocks / num_of_blocks
    return EncodedFrame(mv_field, avg_mae, residual_frame_with_mc, quat_dct_coffs_frame_with_mc, reconstructed_frame_with_mc)

def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple
