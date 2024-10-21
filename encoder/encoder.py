import csv
import time
from contextlib import ExitStack

from skimage.metrics import peak_signal_noise_ratio

from common import pad_frame
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.dct import *
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
        block_size = params.encoder_config.block_size
        search_range = params.encoder_config.search_range


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
            padded__frame = pad_frame(y_plane, block_size)

            if frame_index % params.encoder_config.I_Period == 0:
                frame = IFrame(padded__frame)
            else:
                frame = PFrame(padded__frame, prev_frame)

            frame.encode(params.encoder_config)
            frame.write_metrics_data(metrics_csv_writer, frame_index, params.encoder_config)
            frame.write_encoded_to_file(mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh)
            prev_frame = frame.reconstructed_frame


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


def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple
