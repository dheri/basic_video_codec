import logging
import os

from decoder import decode
from encoder import encode
from metrics import plot_metrics

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(input_file, width, height):

    file_prefix = os.path.splitext(input_file)[0]
    input_file = f'{file_prefix}.y'

    search_ranges = [1, 2, 4, 8]  # Search ranges to test
    block_sizes = [2, 8, 16, 64]  # Block sizes to process
    search_range = search_ranges[1]
    block_size = block_sizes[1]  # Block sizes to process 'i'
    residual_approx_factor = 5
    frames_to_process = 11

    # Assuming a previous frame of all 128 (as specified for the first frame)


    file_identifier = f'{block_size}_{search_range}_{residual_approx_factor}'

    mv_output_file = f'{file_prefix}_{file_identifier}_mv.txt'
    residual_txt_file = f'{file_prefix}_{file_identifier}_residuals.txt'
    residual_yuv_file = f'{file_prefix}_{file_identifier}_residuals.yuv'
    reconstructed_file = f'{file_prefix}_{file_identifier}_recons.yuv'
    decoded_yuv = f'{file_prefix}_{file_identifier}_decoded.yuv'

    if os.path.exists(residual_yuv_file):
        logger.info(f" {residual_yuv_file} already exists. skipping encoding..")
    else:
        avg_mae_per_frame = encode(input_file, mv_output_file, residual_txt_file, residual_yuv_file, reconstructed_file, frames_to_process, height, width, block_size, search_range, residual_approx_factor)
        plot_metrics(avg_mae_per_frame, f'{file_prefix}_{file_identifier}', block_size, search_range)

    decode(residual_yuv_file, mv_output_file, block_size, decoded_yuv, height, width, frames_to_process)


