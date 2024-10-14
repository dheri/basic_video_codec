import logging
import os

from decoder import decode
from encoder import encode
from file_io import get_file_name
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
    search_range = search_ranges[3]
    block_size = block_sizes[1]  # Block sizes to process 'i'
    residual_approx_factor = 3
    frames_to_process = 25

    os.makedirs(os.path.dirname(get_file_name(input_file, '', block_size, search_range, residual_approx_factor)), exist_ok=True)
    mv_txt_file = get_file_name(input_file, 'mv.txt', block_size, search_range, residual_approx_factor)
    residual_yuv_file = get_file_name(input_file, 'residuals.yuv', block_size, search_range, residual_approx_factor)
    reconstructed_file = get_file_name(input_file, 'reconstructed.yuv', block_size, search_range, residual_approx_factor)
    decoded_file = get_file_name(input_file, 'decoded.yuv', block_size, search_range, residual_approx_factor)

    if os.path.exists(residual_yuv_file) and False:
        logger.info(f" {residual_yuv_file} already exists. skipping encoding..")
    else:
        avg_mae_per_frame = encode(input_file, mv_txt_file, residual_yuv_file, reconstructed_file, frames_to_process, height, width, block_size, search_range, residual_approx_factor)
        plot_metrics(avg_mae_per_frame, input_file , block_size, search_range, residual_approx_factor)

    decode(residual_yuv_file, mv_txt_file, block_size, decoded_file, height, width, frames_to_process)


