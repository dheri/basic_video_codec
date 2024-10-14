import logging
import os

from decoder import decode
from encoder import encode
from file_io import FileIOHelper
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
    residual_approx_factor = 3
    frames_to_process = 25

    file_io = FileIOHelper(input_file, block_size, search_range, residual_approx_factor)

    if os.path.exists(file_io.get_mc_residual_file_name()) and False:
        logger.info(f" {file_io.get_mc_residual_file_name()} already exists. skipping encoding..")
    else:
        avg_mae_per_frame = encode(input_file, frames_to_process, height, width, block_size, search_range, residual_approx_factor)
        plot_metrics(file_io , block_size, search_range, residual_approx_factor)

    decode(file_io, block_size, height, width, frames_to_process)


