from contextlib import ExitStack

import numpy as np

from common import get_logger
from encoder.dct import generate_quantization_matrix, rescale_block, apply_idct_2d
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from motion_vector import parse_mv

logger = get_logger()
def find_predicted_block(mv, x, y, prev_frame, block_size):
    # Calculate the predicted block coordinates
    pred_x = x + mv[0]
    pred_y = y + mv[1]

    # Clip the coordinates to ensure they are within bounds
    pred_x = np.clip(pred_x, 0, prev_frame.shape[1] - block_size)
    pred_y = np.clip(pred_y, 0, prev_frame.shape[0] - block_size)

    predicted_block = prev_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
    return predicted_block


def decode_frame(quant_dct_coff_frame, prev_frame, mv_frame, input_parameters: InputParameters):
    block_size = input_parameters.encoder_parameters.block_size
    quantization_factor = input_parameters.encoder_parameters.quantization_factor
    height, width = input_parameters.height, input_parameters.width
    decoded_frame = np.zeros_like(prev_frame, dtype=np.uint8)

    # Generate the quantization matrix Q based on block size and quantization factor
    Q = generate_quantization_matrix(block_size, quantization_factor)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Get the quantized residual block
            dct_coffs_block = quant_dct_coff_frame[y:y + block_size, x:x + block_size]

            # Rescale the residual block by multiplying by Q
            rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)

            # Apply inverse DCT to the rescaled residual block
            idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)

            # Get the predicted block using the motion vector
            predicted_b = find_predicted_block(mv_frame[(x, y)], x, y, prev_frame, block_size).astype(np.int16)

            # Reconstruct the block by adding the predicted block and the rescaled residual
            decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)

            # Clip values to avoid overflow/underflow and convert back to uint8
            decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)

            # Place the reconstructed block in the decoded frame
            decoded_frame[y:y + block_size, x:x + block_size] = decoded_block

    return decoded_frame


def decode(params: InputParameters):
    file_io = FileIOHelper(params)
    frames_to_process = params.frames_to_process
    height = params.height
    width = params.width
    mv_txt_file = file_io.get_mv_file_name()
    decoded_yuv = file_io.get_mc_decoded_file_name()

    frame_size = width * height
    prev_frame = np.full((height, width), 128, dtype=np.uint8)


    with ExitStack() as stack:
        quant_dct_coff_fh = stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'rb'))
        mv_txt_fh = stack.enter_context(open(mv_txt_file, 'rt'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            quant_dct_coff = quant_dct_coff_fh.read(frame_size)
            mv_txt =  mv_txt_fh.readline()
            if not quant_dct_coff or frame_index > frames_to_process or not mv_txt:
                break  # End of file or end of frames
            logger.debug(f"Decoding frame {frame_index}/{frames_to_process}")
            quant_dct_coff_frame = np.frombuffer(quant_dct_coff, dtype=np.uint8).reshape((height, width))
            mv = parse_mv(mv_txt)

            decoded_frame = decode_frame(quant_dct_coff_frame, prev_frame, mv, params)
            write_y_only_frame(decoded_fh, decoded_frame)

            prev_frame = decoded_frame
    print('end decoding')