from collections import deque
from contextlib import ExitStack

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.PredictionMode import PredictionMode
from encoder.entropy_encoder import *
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from common import *

logger = get_logger()

"""
def entropy_decode(bitstream):
    decoded_symbols = []
    while len(bitstream) > 0:
        symbol, bitstream = exp_golomb_decode(bitstream)
        decoded_symbols.append(symbol)
    return decoded_symbols


def decode_video(params: InputParameters):
    file_io = FileIOHelper(params)
    frames_to_process = params.frames_to_process
    height = params.height
    width = params.width
    decoded_yuv = file_io.get_mc_decoded_file_name()

    frame_size = width * height
    reference_frames = deque(maxlen=params.encoder_config.nRefFrames)
    reference_frames.append(np.full((height, width), 128, dtype=np.uint8))

    with ExitStack() as stack:
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        encoded_fh = stack.enter_context(open(file_io.get_encoded_file_name(), 'rb'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            if frame_index > frames_to_process or not encoded_fh:
                break  # End of file or end of frames
            prediction_mode = int.from_bytes(encoded_fh.read(1))
            logger.debug(f"Decoding {PredictionMode(prediction_mode)} {frame_index}/{frames_to_process}")

            if prediction_mode == PredictionMode.INTRA_FRAME.value:
                frame = IFrame()
                reference_frames.clear()
            else:
                frame = PFrame(reference_frames=reference_frames)

            prediction_data_len = int.from_bytes(encoded_fh.read(2))
            prediction_data = encoded_fh.read(prediction_data_len)
            frame.entropy_decode_prediction_data(prediction_data, params)

            ee_dct_coffs_len = int.from_bytes(encoded_fh.read(3))
            frame.entropy_encoded_DCT_coffs = encoded_fh.read(ee_dct_coffs_len)
            frame.entropy_decode_dct_coffs(params)

            # Decode the frame
            decoded_frame = frame.decode_mc_q_dct((params.height, params.width), encoder_config=params.encoder_config)

            # Read and compare with the reconstructed frame
            reconstructed_frame = np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape(
                (height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)
            dct_coffs_extremes = frame.get_quat_dct_coffs_extremes()

            logger.info(
                f"{frame_index:2}: psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")

            # Write the decoded frame to the output file
            write_y_only_frame(decoded_fh, decoded_frame)

            reference_frames.append(decoded_frame)
            # logger.debug(f"reference_frames: {len(reference_frames)}, {reference_frames.maxlen}")
    logger.info('End decoding')
"""

from collections import deque
from contextlib import ExitStack

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.PredictionMode import PredictionMode
from encoder.entropy_encoder import *
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters

logger = get_logger()


def entropy_decode(bitstream):
    decoded_symbols = []
    while len(bitstream) > 0:
        symbol, bitstream = exp_golomb_decode(bitstream)
        decoded_symbols.append(symbol)
    return decoded_symbols


def decode_video(params: InputParameters):
    file_io = FileIOHelper(params)
    frames_to_process = params.frames_to_process
    height = params.height
    width = params.width
    decoded_yuv = file_io.get_mc_decoded_file_name()

    frame_size = width * height
    reference_frames = deque(maxlen=params.encoder_config.nRefFrames)
    reference_frames.append(np.full((height, width), 128, dtype=np.uint8))

    with ExitStack() as stack:
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        encoded_fh = stack.enter_context(open(file_io.get_encoded_file_name(), 'rb'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            if frame_index > frames_to_process or not encoded_fh:
                break  # End of file or end of frames
            prediction_mode = int.from_bytes(encoded_fh.read(1))
            logger.debug(f"Decoding {PredictionMode(prediction_mode)} {frame_index}/{frames_to_process}")

            if prediction_mode == PredictionMode.INTRA_FRAME.value:
                frame = IFrame()
                reference_frames.clear()
            else:
                frame = PFrame(reference_frames=reference_frames)

            prediction_data_len = int.from_bytes(encoded_fh.read(2))
            prediction_data = encoded_fh.read(prediction_data_len)
            frame.entropy_decode_prediction_data(prediction_data, params)

            # Parse the split flag
            is_split = int.from_bytes(encoded_fh.read(1), 'big')
            logger.debug(f"Split flag for frame {frame_index}: {is_split}")

            # Handle sub-block decoding if split is enabled
            if is_split:
                decode_subblocks(frame, encoded_fh, params)
            else:
                ee_dct_coffs_len = int.from_bytes(encoded_fh.read(3))
                frame.entropy_encoded_DCT_coffs = encoded_fh.read(ee_dct_coffs_len)
                frame.entropy_decode_dct_coffs(params)

            # Decode the frame
            decoded_frame = frame.decode_mc_q_dct((params.height, params.width), encoder_config=params.encoder_config)

            # Read and compare with the reconstructed frame
            reconstructed_frame = np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape(
                (height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)
            dct_coffs_extremes = frame.get_quat_dct_coffs_extremes()

            logger.info(
                f"{frame_index:2}: psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")

            # Write the decoded frame to the output file
            write_y_only_frame(decoded_fh, decoded_frame)

            reference_frames.append(decoded_frame)
            # logger.debug(f"reference_frames: {len(reference_frames)}, {reference_frames.maxlen}")
    logger.info('End decoding')


def decode_subblocks(frame, encoded_fh, params):
    """
    Decode sub-blocks in Z-order if the block is split.
    """
    block_size = params.encoder_config.block_size
    sub_block_size = block_size // 2
    height, width = params.height, params.width

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Decode sub-blocks in Z-order
            for sub_y in [y, y + sub_block_size]:
                for sub_x in [x, x + sub_block_size]:
                    dct_coffs_len = int.from_bytes(encoded_fh.read(3))
                    frame.entropy_encoded_DCT_coffs = encoded_fh.read(dct_coffs_len)
                    frame.entropy_decode_dct_coffs(params, sub_block=True, sub_block_coords=(sub_x, sub_y))
