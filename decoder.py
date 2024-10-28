from contextlib import ExitStack

import numpy as np

from common import get_logger
from encoder.IFrame import IFrame
from encoder.PFrame import construct_frame_from_dct_and_mv, PFrame
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from motion_vector import parse_mv
from skimage.metrics import peak_signal_noise_ratio
from encoder.entropy_encoder import *
logger = get_logger()

def entropy_decode(bitstream):
    """Decodes the entropy-encoded bitstream into quantized DCT coefficients."""
    decoded_symbols = []
    while bitstream:
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
    prev_frame = np.full((height, width), 128, dtype=np.uint8)

    # Read the stored frame sizes
    with open('frame_sizes.txt', 'r') as frame_sizes_file:
        frame_sizes = list(map(int, frame_sizes_file.read().split(',')))

    with ExitStack() as stack:
        quant_dct_coff_fh = stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'rb'))
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        encoded_fh = stack.enter_context(open(file_io.get_encoded_file_name(), 'rb'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            quant_dct_coff = quant_dct_coff_fh.read(frame_size * 2)  # quant_dct_coff are stored as int16. i.e. 2 bytes
            if not quant_dct_coff or frame_index > frames_to_process or not encoded_fh:
                break  # End of file or end of frames
            logger.info(f"Decoding frame {frame_index}/{frames_to_process}")
            if (frame_index - 1) % params.encoder_config.I_Period == 0:
                frame = IFrame()
            else:
                frame = PFrame(prev_frame=prev_frame)

            # Use the stored frame size for decoding
            bytes_per_frame = frame_sizes[frame_index - 1] //8




            entropy_encoded_bitstream = encoded_fh.read(bytes_per_frame)
            logger.info(f"reading {bytes_per_frame} bytes for {frame.prediction_mode}")
            encoded_bitstream = entropy_decode(entropy_encoded_bitstream)
            frame.construct_frame_metadata_from_bit_stream(params, encoded_bitstream)

            decoded_frame = frame.decode((params.height, params.width), encoder_config=params.encoder_config)

            reconstructed_frame = np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape((height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)

            logger.info(f"{frame_index:2}: psnr [{round(psnr, 2):6.2f}]")

            write_y_only_frame(decoded_fh, decoded_frame)

            prev_frame = decoded_frame

    logger.info('end decoding')
