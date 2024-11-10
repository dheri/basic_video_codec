import csv
import time
from contextlib import ExitStack

from skimage.metrics import peak_signal_noise_ratio

from common import pad_frame, bytes_to_int_3, int_to_3_bytes
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.dct import *
from file_io import write_mv_to_file, write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from encoder.entropy_encoder import *
from bitarray import bitarray
import numpy as np

logger = get_logger()

def encode_video(params: InputParameters):
    file_io = FileIOHelper(params)

    start_time = time.time()
    y_size = params.width * params.height
    prev_frame = np.full((params.height, params.width), 128, dtype=np.uint8)
    

    with ExitStack() as stack:
        f_in = stack.enter_context(open(params.y_only_file, 'rb'))
        mv_fh = stack.enter_context(open(file_io.get_mv_file_name(), 'wb'))
        quant_dct_coff_fh = stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'wb'))
        residual_w_mc_yuv_fh = stack.enter_context(open(file_io.get_residual_w_mc_file_name(), 'wb'))
        residual_wo_mc_yuv_fh = stack.enter_context(open(file_io.get_residual_wo_mc_file_name(), 'wb'))
        reconstructed_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'wb'))
        encoded_fh = stack.enter_context(open(file_io.get_encoded_file_name(), 'wb'))

        metrics_csv_fh = stack.enter_context(open(file_io.get_metrics_csv_file_name(), 'wt', newline=''))
        frames_to_process = params.frames_to_process
        height = params.height
        width = params.width
        block_size = params.encoder_config.block_size
        search_range = params.encoder_config.search_range
        quantization_factor = params.encoder_config.quantization_factor
        I_Period = params.encoder_config.I_Period

        metrics_csv_writer = csv.writer(metrics_csv_fh)
        # Include QP, I_Period, and total bit size in the CSV header
        metrics_csv_writer.writerow(['Frame Index', 'Average MAE', 'PSNR', 'Encoded Frame Size in Bytes', 'QP', 'I_Period', 'Total Bit Size'])
        frame_index = 0

        while True:
            encoded_fh_idx =encoded_fh.tell()

            frame_index += 1
            y_frame = f_in.read(y_size)
            if not y_frame or frame_index > frames_to_process:
                break  # End of file or end of frames
            # logger.info(f"Encoding frame {frame_index}/{frames_to_process}")
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))
            padded_frame = pad_frame(y_plane, block_size)

            # if (frame_index - 1) % I_Period == 0:
            #     frame = IFrame(padded_frame)
            # else:
            #     frame = PFrame(padded_frame, prev_frame)

            frame = PFrame(padded_frame, prev_frame)

            frame.encode(params.encoder_config)
            # frame.populate_bit_stream_buffer(params.encoder_config)

            # frame.generate_prediction_data()
            frame.entropy_encode_prediction_data()


            frame.entropy_encode_dct_coffs(block_size)
            # 1 byte for prediction_mode
            encoded_fh.write(frame.prediction_mode.value.to_bytes(1))

            # 2 byte for len of entropy_encoded_prediction_data
            num_of_byte_in_entropy_encoded_prediction_data = (len(frame.entropy_encoded_prediction_data) + 7) // 8  # plus 7 to get ceiling of bytes
            # logger.info(f"num_of_byte_in_entropy_encoded_prediction_data  {num_of_byte_in_entropy_encoded_prediction_data.to_bytes(2)}")
            encoded_fh.write(num_of_byte_in_entropy_encoded_prediction_data.to_bytes(2))

            # n bytes for entropy_encoded_prediction_data
            encoded_fh.write(frame.entropy_encoded_prediction_data.tobytes())

            # 3 byte for len of entropy_encoded_DCT_coffs
            num_of_byte_in_entropy_encoded_dct_coffs = (len(frame.entropy_encoded_DCT_coffs) + 7) // 8  # plus 7 to get ceiling of bytes
            # logger.info(f"num_of_byte_in_entropy_encoded_prediction_data  {num_of_byte_in_entropy_encoded_dct_coffs.to_bytes(3)}")
            encoded_fh.write(num_of_byte_in_entropy_encoded_dct_coffs.to_bytes(3))

            # n bytes for entropy_encoded_DCT_coffs
            encoded_fh.write(frame.entropy_encoded_DCT_coffs.tobytes())


            # Calculate PSNR and MAE
            frame_psnr = peak_signal_noise_ratio(frame.curr_frame, frame.reconstructed_frame)
            mae = frame.avg_mae

            encoded_frame_size = encoded_fh.tell() -  encoded_fh_idx
            metrics_csv_writer.writerow([frame_index, mae, frame_psnr, encoded_frame_size, quantization_factor, I_Period, ])


            dct_coffs_extremes = frame.get_quat_dct_coffs_extremes()
            frame_info_str = (f"{frame_index:2}: {params.encoder_config.block_size} {params.encoder_config.search_range} {params.encoder_config.quantization_factor}, mae [{round(frame.avg_mae, 2):7.2f}] psnr [{round(frame_psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4},"
                              f" {dct_coffs_extremes[1]:3}] SOB [{encoded_fh_idx}] size: [{encoded_frame_size}] ")
            logger.info(frame_info_str)

            frame.write_encoded_to_file( mv_fh, quant_dct_coff_fh, residual_w_mc_yuv_fh, residual_wo_mc_yuv_fh, reconstructed_fh, params.encoder_config)

            prev_frame = frame.reconstructed_frame

    end_time = time.time()
    elapsed_time = end_time - start_time

    num_of_blocks = (height // block_size) * (width // block_size)
    num_of_comparisons = num_of_blocks * (2 * search_range + 1) ** 2
    result = str(f"{num_of_comparisons / elapsed_time:9.3f} | {num_of_comparisons:7d} | {num_of_blocks / elapsed_time:7.3f} |  {num_of_blocks:5d} | {frames_to_process / elapsed_time:6.2f} | {frames_to_process:3d} | {elapsed_time:6.3f} | {block_size:2d} | {search_range:2d} |\n")
    logger.info(result)
    
    with open('results.csv', 'at') as f_in:
        f_in.write(result)
    logger.debug('end encoding')
    return

def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple
