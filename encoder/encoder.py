import csv
import os
import time
from collections import deque
from contextlib import ExitStack
from copy import copy

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from common import pad_frame
from encoder.Frame import Frame
from encoder.FrameMetrics import FrameMetrics
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.PredictionMode import PredictionMode
from encoder.RateControl.RateControl import bit_budget_per_frame
from encoder.RateControl.lookup import rc_lookup_file_path, get_combined_lookup_table
from encoder.block_predictor import build_pre_interpolated_buffer
from encoder.dct import *
from encoder.entropy_encoder import *
from file_io import FileIOHelper
from input_parameters import InputParameters

logger = get_logger()


def encode_video(params: InputParameters):
    file_io = FileIOHelper(params)
    scene_change_threshold = 1.3
    y_size = params.width * params.height

    reference_frames = deque(maxlen=params.encoder_config.nRefFrames)
    reference_frames.append(np.full((params.height, params.width), 128, dtype=np.uint8))

    interpolated_reference_frames = deque(maxlen=params.encoder_config.nRefFrames)
    interpolated_reference_frames.append(build_pre_interpolated_buffer(reference_frames[0]))

    if params.encoder_config.RCflag or 1:
        rc_lookup_file_path_i = rc_lookup_file_path(params.encoder_config, 'I' )
        rc_lookup_file_path_p = rc_lookup_file_path(params.encoder_config, 'P' )
        params.encoder_config.rc_lookup_table = get_combined_lookup_table(rc_lookup_file_path_i, rc_lookup_file_path_p)


    with ExitStack() as stack:
        f_in = stack.enter_context(open(params.y_only_file, 'rb'))
        mv_fh = stack.enter_context(open(file_io.get_mv_file_name(), 'wt'))
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
        I_Period = params.encoder_config.I_Period
        metrics_csv_writer = csv.writer(metrics_csv_fh)
        metrics_csv_writer.writerow(FrameMetrics.get_header())
        frame_index = 0
        logger.info(
            f"[i={params.encoder_config.block_size}"
            f" r={params.encoder_config.search_range}"
            f" q={params.encoder_config.quantization_factor}]"
            f" nRefFrames=[{params.encoder_config.nRefFrames}]"
            f" fracMeEnabled=[{params.encoder_config.fracMeEnabled}]"
            f" RateControl=[{params.encoder_config.RCflag}] @ [{params.encoder_config.targetBR} _bps]"
        )
        prev_frame = Frame()
        prev_frame.rc_qp_per_row = [params.encoder_config.quantization_factor] #arbitatry qp seeding for first frame
        video_enc_start_time = time.time()
        while True:
            frame_enc_start_time = time.time()
            start_of_bock_idx = encoded_fh.tell()
            frame_index += 1
            y_frame = f_in.read(y_size)
            if not y_frame or frame_index > frames_to_process:
                break  # End of file or end of frames
            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))
            padded_frame = pad_frame(y_plane, block_size)

            first_pass_frame = get_first_pass_frame(frame_index, interpolated_reference_frames, padded_frame, reference_frames, params, prev_frame)
            first_pass_frame.encode_mc_q_dct(params.encoder_config)
            frame = first_pass_frame

            pframe_overage = first_pass_frame.get_overage_ratios(params.encoder_config)
            # Second pass
            if params.encoder_config.RCflag > 1:
                is_scene_change = False
                if first_pass_frame.is_pframe() and pframe_overage[1] > scene_change_threshold:
                    frame.scaling_factor = (1 - pframe_overage[1])  * 0.95
                    logger.info(f"scene change detected in pframe:  {sum(first_pass_frame.bits_per_row)} {pframe_overage[0]:4.2f} | {pframe_overage[1]:4.2f}")
                    is_scene_change = True
                frame = get_second_pass_frame(padded_frame, reference_frames, interpolated_reference_frames, params, first_pass_frame, is_scene_change, prev_frame)
                frame.encode_mc_q_dct(params.encoder_config)


            # frame in encoded, rest is writing data to file frame_enc_time
            frame_enc_time = time.time() - frame_enc_start_time

            # 1 byte for prediction_mode
            encoded_fh.write(frame.prediction_mode.value.to_bytes(1))

            # 2 byte for len of entropy_encoded_prediction_data
            ee_prediction_data_byte_size = (len(frame.entropy_encoded_prediction_data) + 7) // 8  # +7 for ceiling of bytes
            encoded_fh.write(ee_prediction_data_byte_size.to_bytes(2))

            # n bytes for entropy_encoded_prediction_data
            start_of_prediction_data_idx = encoded_fh.tell()
            encoded_fh.write(frame.entropy_encoded_prediction_data.tobytes())

            # 3 byte for len of entropy_encoded_DCT_coffs
            ee_dct_coffs_byte_size = (len(frame.entropy_encoded_DCT_coffs) + 7) // 8  # + 7 for ceiling of bytes
            encoded_fh.write(ee_dct_coffs_byte_size.to_bytes(3))

            # n bytes for entropy_encoded_DCT_coffs
            start_of_dct_coffs_idx = encoded_fh.tell()
            encoded_fh.write(frame.entropy_encoded_DCT_coffs.tobytes())

            frame_psnr = peak_signal_noise_ratio(frame.curr_frame, frame.reconstructed_frame)
            dct_coffs_extremes = frame.get_quat_dct_coffs_extremes()
            mv_extremes = frame.get_mv_extremes()

            encoded_frame_size = encoded_fh.tell() - start_of_bock_idx
            frame_metrics = FrameMetrics(
                frame_index, frame.prediction_mode.value,
                frame.avg_mae, frame.total_mae_comparisons,
                frame_psnr, encoded_frame_size, encoded_fh.tell() * 8,
                frame_enc_time, (time.time() - video_enc_start_time))

            metrics_csv_writer.writerow(frame_metrics.to_csv_row())


            frame_info_str = (
                f"{frame_index:2}: {frame.prediction_mode} "
                f" mae [{round(frame.avg_mae, 2):6.2f}] "
                # f"mv_extremes: [{mv_extremes[0]}, {mv_extremes[1]}] "
                # f" mae_comps [{round(frame.total_mae_comparisons, 2):7d}] "
                f"psnr [{round(frame_psnr, 2):6.2f}], "
                f"q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}] "
                f"size: [{encoded_frame_size:6}] | [{(sum(frame.bits_per_row) + 7) // 8 :6}] diff: {(sum(frame.bits_per_row)+7)//8-encoded_frame_size + 6:4.2f} "
                # f"SoB [{hex(start_of_bock_idx):7}] "
                # f"SoPD [{hex(start_of_prediction_data_idx):7}] "
                # f"SoDCTCoffs [{hex(start_of_dct_coffs_idx):7}] "
            )
            logger.info(frame_info_str)

            frame.write_encoded_to_file(mv_fh, quant_dct_coff_fh, residual_w_mc_yuv_fh, residual_wo_mc_yuv_fh,
                                        reconstructed_fh, params.encoder_config)

            reference_frames.append(frame.reconstructed_frame)
            interpolated_reference_frames.append(build_pre_interpolated_buffer(frame.reconstructed_frame))
            prev_frame = frame


    end_time = time.time()
    elapsed_time = end_time - video_enc_start_time

    num_of_blocks = (height // block_size) * (width // block_size)
    num_of_comparisons = num_of_blocks * (2 * search_range + 1) ** 2
    result = str(
        f"{num_of_comparisons / elapsed_time:9.3f} | {num_of_comparisons:7d} | {num_of_blocks / elapsed_time:7.3f} |  {num_of_blocks:5d} | {frames_to_process / elapsed_time:6.2f} | {frames_to_process:3d} | {elapsed_time:6.3f} | {block_size:2d} | {search_range:2d} |\n")
    logger.info(result)
    results_file_name = os.path.join(os.path.dirname(__file__), f'../results.csv')
    with open(results_file_name, 'at') as f_in:
        f_in.write(result)
    logger.debug('end encoding')
    return


def get_first_pass_frame(frame_index, interpolated_reference_frames, padded_frame, reference_frames, params: InputParameters, prev_frame:Frame):
    if (frame_index - 1) % params.encoder_config.I_Period == 0:
        first_pass_frame = IFrame(padded_frame)
        reference_frames.clear()
        interpolated_reference_frames.clear()
    else:
        first_pass_frame = PFrame(padded_frame, reference_frames, interpolated_reference_frames)
    first_pass_frame.is_first_pass = True
    first_pass_frame.prev_frame = prev_frame

    first_pass_frame.index = frame_index
    first_pass_frame.bit_budget = bit_budget_per_frame(params.encoder_config)
    return first_pass_frame

def get_second_pass_frame(padded_frame, reference_frames, interpolated_reference_frames, params:InputParameters, first_pass_frame: Frame, is_scene_change=False, prev_frame:Frame=None):
    if is_scene_change or first_pass_frame.is_iframe():
        reference_frames.clear()
        interpolated_reference_frames.clear()
        frame = IFrame(padded_frame)
    else:
        frame = PFrame(padded_frame, reference_frames, interpolated_reference_frames)
    frame.is_first_pass = False
    frame.prev_frame = prev_frame

    frame.index = first_pass_frame.index
    frame.bit_budget = bit_budget_per_frame(params.encoder_config)
    frame.prev_pass_frame = first_pass_frame
    return frame


def round_to_nearest_multiple(arr, n):
    multiple = 2 ** n
    return np.round(arr / multiple) * multiple

