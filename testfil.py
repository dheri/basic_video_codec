import numpy as np

from common import mae
import struct

import math

import numpy as np
from scipy.fftpack import dct, idct

from common import get_logger
from encoder.params import validate_qp
import concurrent
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from common import calculate_num_frames, pad_frame, split_into_blocks
from file_io import FileIOHelper
from input_parameters import InputParameters
from main import logger


import csv

from matplotlib import pyplot as plt

from file_io import FileIOHelper
from input_parameters import InputParameters


import os

from input_parameters import InputParameters

import os
import numpy as np
from scipy.ndimage import zoom
from common import calculate_num_frames
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from typing import Self

from common import get_logger, generate_residual_block, find_predicted_block
from encoder.Frame import Frame, PredictionMode
from encoder.block_predictor import predict_block
from encoder.dct import apply_dct_2d, generate_quantization_matrix, quantize_block, rescale_block, apply_idct_2d
from encoder.params import EncoderConfig, EncodedPBlock
from concurrent import futures

from file_io import write_mv_to_file, write_y_only_frame
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
from contextlib import ExitStack

import numpy as np

from common import get_logger
from encoder.IFrame import IFrame
from encoder.PFrame import decode_p_frame, PFrame
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from motion_vector import parse_mv
from skimage.metrics import peak_signal_noise_ratio

import math

import numpy as np

from enum import Enum
from typing import Optional

import numpy as np

from encoder.byte_stream_buffer import BitStreamBuffer
from encoder.params import EncoderConfig

import numpy as np

from encoder.Frame import Frame, PredictionMode
from encoder.dct import apply_dct_2d, generate_quantization_matrix, quantize_block, rescale_block, apply_idct_2d
from encoder.params import EncoderConfig

import numpy as np
def intra_predict_block(curr_block, reconstructed_frame, x, y, block_size):
    # Horizontal predictor
    if x > 0:  # Not on the left border
        left_samples = reconstructed_frame[y:y + block_size, x - 1]  # Left i-samples
        horizontal_pred = np.tile(left_samples, (block_size, 1))  # Repeat left samples horizontally
    else:
        horizontal_pred = np.full((block_size, block_size), 128)  # Use value 128 for border

    # Vertical predictor
    if y > 0:  # Not on the top border
        top_samples = reconstructed_frame[y - 1, x:x + block_size]  # Top i-samples
        vertical_pred = np.tile(top_samples, (block_size, 1)).T  # Repeat top samples vertically
    else:
        vertical_pred = np.full((block_size, block_size), 128)  # Use value 128 for border

    # Calculate MAE for both modes
    mae_horizontal = np.mean(np.abs(curr_block - horizontal_pred))
    mae_vertical = np.mean(np.abs(curr_block - vertical_pred))

    # Select the mode with the lowest MAE
    if mae_horizontal < mae_vertical:
        return horizontal_pred, 0  # Horizontal mode (0)
    else:
        return vertical_pred, 1  # Vertical mode (1)

def differential_encode_mode(current_mode, previous_mode):
    return current_mode - previous_mode

def differential_decode_mode(diff_mode, previous_mode):
    return diff_mode + previous_mode


def predict_block(curr_block, prev_partial_frame, block_size):
    return find_lowest_mae_block(curr_block, prev_partial_frame, block_size)

def find_lowest_mae_block(curr_block, prev_partial_frame, block_size):
    """Find the block with the lowest MAE from a smaller previous partial frame."""
    height, width = prev_partial_frame.shape
    if width < block_size or height < block_size:
        raise ValueError(f"width [{width}] or height [{height}] of given block  < block_size [{block_size}]")
    min_mae = float('inf')
    best_mv = [0,0]  # motion vector wrt origin of prev_partial_frame

    # Loop through all possible positions in the previous partial frame
    ref_block = None

    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            # Update best match if a lower MAE is found, breaking ties as described
            if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = [ref_x , ref_y]  # (dx, dy)


    return best_mv, min_mae, ref_block





class BitStreamBuffer:
    def __init__(self):
        self.byte_stream : bytearray = bytearray()  # To hold the bitstream in bytes
        self.current_byte = 0  # To accumulate bits into a byte
        self.bit_position = 0  # Current bit position in the current_byte

    def write_bit(self, bit):
        """Write a single bit to the buffer."""
        if bit not in [0, 1]:
            raise ValueError("Bit value must be 0 or 1")

        # Add the bit to the current_byte at the current position
        self.current_byte = (self.current_byte << 1) | bit
        self.bit_position += 1

        # If a byte (8 bits) is filled, append it to the byte stream
        if self.bit_position == 8:
            self.byte_stream.append(self.current_byte)
            self.current_byte = 0
            self.bit_position = 0

    def write_bits(self, value, num_bits):
        """Write a value as a series of bits."""
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.write_bit(bit)

    def write_int16(self, value):
        """Write a 16-bit signed integer to the bitstream."""
        # Ensure the value fits within int16 range
        if not (-32768 <= value <= 32767):
            raise OverflowError("Value out of range for int16.")
        # Write high byte and low byte (big-endian)
        self.write_bits(value >> 8 & 0xFF, 8)  # High byte
        self.write_bits(value & 0xFF, 8)  # Low byte

    def write_quantized_coeffs(self, coeffs_2d):
        """Write a 2D array of quantized coefficients to the buffer."""
        flat_coeffs = coeffs_2d.flatten()
        for coeff in flat_coeffs:
            self.write_int16(coeff)

    def read_bit(self):
        """Read the next bit from the buffer."""
        if self.bit_position == 0:
            if not self.byte_stream:
                raise EOFError("No more bits to read")
            # Load the next byte and reset bit position
            self.current_byte = self.byte_stream.pop(0)
            self.bit_position = 8

        # Read the next bit
        self.bit_position -= 1
        return (self.current_byte >> self.bit_position) & 1

    def read_bits(self, num_bits):
        """Read a specified number of bits from the buffer."""
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value

    def read_int16(self):
        """Read a 16-bit signed integer from the buffer."""
        high_byte = self.read_bits(8)
        low_byte = self.read_bits(8)
        value = (high_byte << 8) | low_byte
        # Handle negative numbers
        if value >= 0x8000:
            value -= 0x10000
        return value

    def read_quantized_coeffs(self, num_coeffs):
        """Read a specified number of 16-bit quantized coefficients from the buffer."""
        coeffs = []
        for _ in range(num_coeffs):
            coeffs.append(self.read_int16())
        return np.array(coeffs).reshape(int(np.sqrt(num_coeffs)), int(np.sqrt(num_coeffs)))

    def flush(self):
        """Flush remaining bits in the buffer to the byte stream."""
        if self.bit_position > 0:
            self.current_byte <<= (8 - self.bit_position)  # Align the remaining bits to the left
            self.byte_stream.append(self.current_byte)
            self.current_byte = 0
            self.bit_position = 0

    def get_bitstream(self):
        """Return the byte stream."""
        return self.byte_stream

    def __repr__(self):
        bin_rep =  ''.join(f'{byte:08b}' for byte in bytes(self.byte_stream))
        return f"hex:\t{self.byte_stream.hex()} \nbin:\t{bin_rep}"

def compare_bits(byte_array, byte_index, bit_index, expected_value):
    """
    Compares the bit at a specific position in a byte array to an expected value.

    :param byte_array: The byte array.
    :param byte_index: The index of the byte in the array.
    :param bit_index: The index of the bit in the byte (0 is the least significant bit).
    :param expected_value: The expected value of the bit (0 or 1).
    :return: True if the bit matches the expected value, False otherwise.
    """
    if bit_index < 0 or bit_index > 7:
        raise ValueError("bit_index should be between 0 and 7")

    byte = byte_array[byte_index]
    mask = 1 << bit_index
    bit_value = (byte & mask) >> bit_index

    return bit_value == expected_value



logger = get_logger()


def apply_dct_2d(block):
    """Applies 2D DCT to a block using separable 1D DCT."""
    block = block.astype(np.float32)
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct_2d(block):
    """Applies 2D Inverse DCT to a block."""
    block = block.astype(np.float32)
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def generate_quantization_matrix(i, qp):
    """Generates the quantization matrix Q for a given block size i and quantization parameter QP."""
    Q = np.zeros((i, i), dtype=np.uint16)
    for x in range(i):
        for y in range(i):
            if (x + y) < (i - 1):
                Q[x, y] = 2 ** qp
            elif (x + y) == (i - 1):
                Q[x, y] = 2 ** (qp + 1)
            else:
                Q[x, y] = 2 ** (qp + 2)
    return Q

def quantize_block(dct_block, Q):
    """Quantizes a block by dividing by Q and rounding."""
    return np.round(dct_block / Q)

def rescale_block(quantized_block, Q):
    """Rescales the quantized block by multiplying by Q."""
    return quantized_block * Q


def transform_quantize_rescale_inverse(block, qp):
    """Applies the full pipeline: DCT -> Quantization -> Rescale -> IDCT."""
    i = block.shape[0]  # Block size (e.g., 8 for an 8x8 block)

    # logger.info(f'block: \n{np.ceil(block)}')
    validate_qp(i, qp)

    # Step 1: Apply 2D DCT to the block
    dct_coffs = apply_dct_2d(block)
    # logger.info(f'dct_block: \n{np.ceil(dct_block)}')

    # Step 2: Generate the quantization matrix based on block size and QP
    Q = generate_quantization_matrix(i, qp)
    # logger.info(f'Q: \n{np.ceil(Q)}')

    # Step 3: Quantize the DCT coefficients
    quantized_dct_coffs = quantize_block(dct_coffs, Q)
    # logger.info(f'quantized_block: \n{np.ceil(quantized_block)}')

    # Step 4: Rescale the quantized block by multiplying by Q
    rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
    # logger.info(f'rescaled_block: \n{np.ceil(rescaled_block)}')

    # Step 5: Apply Inverse DCT to reconstruct the block
    reconstructed_block = apply_idct_2d(rescaled_dct_coffs)
    # logger.info(f'reconstructed_block: \n{np.ceil(reconstructed_block)}')

    return reconstructed_block






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



def exp_golomb_encode(value):
    if value == 0:
        return '0'
    sign = -1 if value < 0 else 1
    value = abs(value)
    m = int(np.log2(value + 1))
    prefix = '0' * m + '1'
    suffix = format(value - (1 << m), f'0{m}b')
    return prefix + suffix if sign == 1 else prefix + '1' + suffix

def exp_golomb_decode(bitstream):
    m = 0
    while bitstream[m] == '0':
        m += 1
    value = (1 << m) + int(bitstream[m + 1:], 2)
    return value
def rle_encode(coeffs):
    encoded = []
    i = 0
    while i < len(coeffs):
        if coeffs[i] == 0:
            zero_count = 0
            while i < len(coeffs) and coeffs[i] == 0:
                zero_count += 1
                i += 1
            encoded.append(zero_count)  # Positive for run of zeros
        else:
            nonzero_count = 0
            start_idx = i
            while i < len(coeffs) and coeffs[i] != 0:
                nonzero_count += 1
                i += 1
            encoded.append(-nonzero_count)  # Negative for run of non-zeros
            encoded.extend(coeffs[start_idx:i])
    encoded.append(0)  # End of block
    return encoded



class PredictionMode(Enum):
    INTER_FRAME = 0  # P-frame
    INTRA_FRAME = 1  # I-frame


class FrameHeader:
    """Class to handle frame header information such as frame type and size."""
    def __init__(self, frame_type: PredictionMode, size: int):

        self.size = size



class Frame:
    def __init__(self, curr_frame=None, prev_frame=None, ):
        self.bitstream_buffer : Optional[BitStreamBuffer] = None
        self.prev_frame = prev_frame
        self.curr_frame = curr_frame
        self.prediction_mode: PredictionMode = PredictionMode.INTER_FRAME
        self.residual_frame = None
        self.quantized_dct_residual_frame = None
        self.reconstructed_frame = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_metrics_data(self, metrics_csv_writer, frame_index : int, encoder_config : EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh):
        pass

    def pre_entropy_encoded_frame_bit_stream(self) -> BitStreamBuffer:
        self.bitstream_buffer = BitStreamBuffer()
        self.bitstream_buffer.write_bit(self.prediction_mode.value)
        if self.quantized_dct_residual_frame is not None:
            self.bitstream_buffer.write_quantized_coeffs(self.quantized_dct_residual_frame)
        else:
            raise ValueError("Illegal operation: quantized_dct_residual_frame is None")


        return self.bitstream_buffer

    def read_entropy_encoded_frame_bit_stream(self) -> BitStreamBuffer:
        pass

    def encoded_frame_data_length(self):
        length = 0
        # for  self.prediction_mode.value
        length += 1

        # for  self.prediction_mode.value
        length += self.quantized_dct_residual_frame.size * 16

        # padding
        length += 8 - (length % 8)
        return length

    def get_quat_dct_coffs_extremes(self):
        # Ensure quat_dct_coffs_with_mc is a numpy array to use numpy's min/max
        if isinstance(self.quantized_dct_residual_frame, np.ndarray):
            min_value = np.min(self.quantized_dct_residual_frame)
            max_value = np.max(self.quantized_dct_residual_frame)
            return [min_value, max_value]
        else:
            raise TypeError("quantized_dct_residual_frame must be a numpy array")



class IFrame(Frame):
    def __init__(self, curr_frame=None ):
        super().__init__(curr_frame)
        self.intra_modes = None
        self.prediction_mode = PredictionMode.INTER_FRAME

    def encode(self, encoder_config: EncoderConfig):
        curr_frame = self.curr_frame
        block_size = encoder_config.block_size
        height, width = curr_frame.shape

        mae_of_blocks = 0
        intra_modes = []  # To store the intra prediction modes (0 for horizontal, 1 for vertical)
        reconstructed_frame = np.zeros_like(curr_frame)
        quantized_dct_residual_frame = np.zeros_like(curr_frame, dtype=np.int16)

        # Loop through each block in the frame
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                curr_block = curr_frame[y:y + block_size, x:x + block_size]

                # Process the block
                mode, mae, reconstructed_block, quantized_dct_residual_block, residual_block = process_block(
                    curr_block, reconstructed_frame, x, y, block_size, encoder_config.quantization_factor
                )

                # Store intra mode and update MAE
                intra_modes.append(mode)
                mae_of_blocks += mae

                # Update reconstructed frame and quantized residuals
                reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block
                quantized_dct_residual_frame[y:y + block_size, x:x + block_size] = quantized_dct_residual_block

        avg_mae = mae_of_blocks / ((height // block_size) * (width // block_size))
        self.reconstructed_frame = reconstructed_frame
        self.quantized_dct_residual_frame = quantized_dct_residual_frame
        self.intra_modes = intra_modes
        self.avg_mae = avg_mae





def intra_predict_block(curr_block, reconstructed_frame, x, y, block_size):
    """Predict the block using horizontal and vertical intra prediction, and choose the best mode based on MAE."""
    horizontal_pred = horizontal_intra_prediction(reconstructed_frame, x, y, block_size)
    vertical_pred = vertical_intra_prediction(reconstructed_frame, x, y, block_size)

    mae_horizontal = np.mean(np.abs(curr_block - horizontal_pred))
    mae_vertical = np.mean(np.abs(curr_block - vertical_pred))

    if mae_horizontal < mae_vertical:
        return horizontal_pred, 0, mae_horizontal  # 0 for horizontal mode
    else:
        return vertical_pred, 1, mae_vertical  # 1 for vertical mode


def horizontal_intra_prediction(reconstructed_frame, x, y, block_size):
    """Perform horizontal intra prediction using the left border samples."""
    if x > 0:
        left_samples = reconstructed_frame[y:y + block_size, x - 1]
        return np.tile(left_samples, (block_size, 1))
    else:
        return np.full((block_size, block_size), 128)  # Use 128 for border


def vertical_intra_prediction(reconstructed_frame, x, y, block_size):
    """Perform vertical intra prediction using the top border samples."""
    if y > 0:
        top_samples = reconstructed_frame[y - 1, x:x + block_size]
        return np.tile(top_samples, (block_size, 1)).T
    else:
        return np.full((block_size, block_size), 128)  # Use 128 for border


def process_block(curr_block, reconstructed_frame, x, y, block_size, quantization_factor):
    """Process a block, apply intra prediction, DCT, quantization, and reconstruction."""
    predicted_block, mode, mae = intra_predict_block(curr_block, reconstructed_frame, x, y, block_size)

    # Compute the residual
    residual_block = curr_block.astype(np.int16) - predicted_block.astype(np.int16)

    # Apply DCT
    dct_residual_block = apply_dct_2d(residual_block)

    # Quantization
    Q = generate_quantization_matrix(block_size, quantization_factor)
    quantized_dct_residual_block = quantize_block(dct_residual_block, Q)

    # Inverse quantization and IDCT
    dequantized_dct_residual_block = rescale_block(quantized_dct_residual_block, Q)
    reconstructed_residual_block = apply_idct_2d(dequantized_dct_residual_block)

    # Reconstruct the block
    reconstructed_block = np.round(predicted_block + reconstructed_residual_block).astype(np.uint8)

    return mode, mae, reconstructed_block, quantized_dct_residual_block, residual_block



class EncoderConfig:
    def __init__(self, block_size, search_range, I_Period , quantization_factor):
        validate_qp(block_size, quantization_factor)
        self.block_size = block_size
        self.search_range = search_range
        self.quantization_factor = quantization_factor
        self.I_Period = I_Period
        self.residual_approx_factor = 0



def validate_qp(i,qp):
    if qp > (math.log2(i) + 7):
        raise ValueError(f" qp [{qp}] > {(math.log2(i) + 7)}")

# class EncodedPFrame:
#     def __init__(self, mv_field, avg_mae, residual_frame_with_mc, quat_dct_coffs_with_mc, reconstructed_frame_with_mc):
#         self.mv_field = mv_field
#         self.avg_mae = avg_mae
#         self.residual_frame_with_mc = residual_frame_with_mc
#         self.quat_dct_coffs_with_mc = quat_dct_coffs_with_mc
#         self.reconstructed_frame_with_mc = reconstructed_frame_with_mc
#
#     def get_quat_dct_coffs_extremes(self):
#         # Ensure quat_dct_coffs_with_mc is a numpy array to use numpy's min/max
#         if isinstance(self.quat_dct_coffs_with_mc, np.ndarray):
#             min_value = np.min(self.quat_dct_coffs_with_mc)
#             max_value = np.max(self.quat_dct_coffs_with_mc)
#             return [min_value, max_value]
#         else:
#             raise TypeError("quat_dct_coffs_with_mc must be a numpy array")


class EncodedBlock:
    def __init__(self, block_coords, motion_vector, mae, quantized_dct_coffs, reconstructed_residual_block, reconstructed_block_with_mc ):
        self.block_coords = block_coords
        self.motion_vector = motion_vector
        self.mae = mae
        self.quantized_dct_coffs = quantized_dct_coffs
        self.reconstructed_residual_block = reconstructed_residual_block
        self.reconstructed_block_with_mc = reconstructed_block_with_mc




logger = get_logger()



class PFrame(Frame):
    def __init__(self, curr_frame=None, prev_frame=None):
        super().__init__(curr_frame, prev_frame)
        self.prediction_mode = PredictionMode.INTRA_FRAME
        self.mv_field = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig) -> Self :
        block_size = encoder_config.block_size
        search_range = encoder_config.search_range
        quantization_factor = encoder_config.quantization_factor

        height, width = self.curr_frame.shape
        num_of_blocks = (height // block_size) * (width // block_size)
        mv_field = {}
        mae_of_blocks = 0

        # Initialize output frames
        reconstructed_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.uint8)
        residual_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        quat_dct_coffs_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int16)

        # Process blocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futuress = [executor.submit(self.process_block,  x, y, block_size, search_range, quantization_factor, width, height, mv_field)
                       for y in range(0, height, block_size)
                       for x in range(0, width, block_size)]

            for f in concurrent.futures.as_completed(futuress):
                encoded_block = f.result()
                block_cords = encoded_block.block_coords
                x, y = block_cords

                # Update frames with the encoded block data
                reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
                quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

                mae_of_blocks += encoded_block.mae

        avg_mae = mae_of_blocks / num_of_blocks

        self.mv_field = mv_field
        self.avg_mae = avg_mae
        self.residual_frame = residual_frame_with_mc
        self.quantized_dct_residual_frame = quat_dct_coffs_frame_with_mc
        self.reconstructed_frame = reconstructed_frame_with_mc
        return self


    def process_block(self, x, y, block_size, search_range, quantization_factor, width, height, mv_field):
        curr_block = self.curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)

        # Get motion vector and MAE
        motion_vector, best_match_mae = self.get_motion_vector(curr_block, x, y, block_size, search_range, width, height)
        mv_field[(x, y)] = motion_vector

        # Generate residual and predicted block
        predicted_block_with_mc, residual_block_with_mc = generate_residual_block(curr_block, self.prev_frame, motion_vector, x, y, block_size)

        # Apply DCT and quantization
        quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

        # Reconstruct the block using the predicted and inverse DCT
        clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc)

        return EncodedPBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block, clipped_reconstructed_block)


    def get_motion_vector(self, curr_block, x, y, block_size, search_range, width, height):
        prev_partial_frame_x_start_idx = max(x - search_range, 0)
        prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)
        prev_partial_frame_y_start_idx = max(y - search_range, 0)
        prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)

        prev_partial_frame = self.prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                                        prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

        best_mv_within_search_window, best_match_mae, best_match_block = predict_block(curr_block, prev_partial_frame, block_size)

        motion_vector = [best_mv_within_search_window[0] + prev_partial_frame_x_start_idx - x,
                         best_mv_within_search_window[1] + prev_partial_frame_y_start_idx - y]

        return motion_vector, best_match_mae

    def decode(self, encoder_config: EncoderConfig):
        decode_p_frame(self.quantized_dct_residual_frame, self.prev_frame, self.mv_field, encoder_config)
        pass

    def write_metrics_data(self, metrics_csv_writer, frame_index, encoder_config: EncoderConfig):
        psnr = peak_signal_noise_ratio(self.curr_frame, self.reconstructed_frame)
        dct_coffs_extremes = self.get_quat_dct_coffs_extremes()
        logger.info(
            f"{frame_index:2}: i={encoder_config.block_size} r={encoder_config.search_range}, qp={encoder_config.quantization_factor}, , mae [{round(self.avg_mae, 2):7.2f}] psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")
        metrics_csv_writer.writerow([frame_index, self.avg_mae, psnr])

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh):
        write_mv_to_file(mv_fh, self.mv_field)
        write_y_only_frame(reconstructed_fh, self.reconstructed_frame)
        write_y_only_frame(residual_yuv_fh, self.residual_frame)
        write_y_only_frame(quant_dct_coff_fh, self.quantized_dct_residual_frame)


def apply_dct_and_quantization(residual_block, block_size, quantization_factor):
    dct_coffs = apply_dct_2d(residual_block)
    Q = generate_quantization_matrix(block_size, quantization_factor)
    quantized_dct_coffs = quantize_block(dct_coffs, Q)
    return quantized_dct_coffs, Q


def reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc):
    rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
    idct_residual_block = apply_idct_2d(rescaled_dct_coffs)
    reconstructed_block_with_mc = np.round(idct_residual_block + predicted_block_with_mc).astype(np.int16)
    clipped_reconstructed_block = np.clip(reconstructed_block_with_mc, 0, 255).astype(np.uint8)
    return clipped_reconstructed_block, idct_residual_block


def decode_p_frame(quant_dct_coff_frame, prev_frame, mv_frame, encoder_config: EncoderConfig):
    block_size = encoder_config.block_size
    quantization_factor = encoder_config.quantization_factor
    height, width = quant_dct_coff_frame.shape
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


import logging
import os

import numpy as np


def get_logger():
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%H:%M:%S', )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def calculate_num_frames(file_path, width, height):
    file_size = os.path.getsize(file_path)
    frame_size = width * height + 2 * (width // 2) * (height // 2)
    return file_size // frame_size

def pad_frame(frame, block_size, pad_value=128):
    height, width = frame.shape
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        padded_frame = np.full((height + pad_height, width + pad_width), pad_value, dtype=np.uint8)
        padded_frame[:height, :width] = frame
        return padded_frame
    return frame

# Function to split the frame into blocks of size (block_size x block_size)
def split_into_blocks(frame, block_size):
    height, width = frame.shape
    return (frame.reshape(height // block_size, block_size, -1, block_size)
                 .swapaxes(1, 2)
                 .reshape(-1, block_size, block_size))

def mae(block1, block2):
    """Compute Mean Absolute Error between two blocks."""
    return np.mean(np.abs(block1 - block2))


def generate_residual_block(curr_block, prev_frame, motion_vector, x, y, block_size):
    predicted_block_with_mc = find_predicted_block(motion_vector, x, y, prev_frame, block_size).astype(np.int16)
    residual_block_with_mc = np.subtract(curr_block, predicted_block_with_mc)
    return predicted_block_with_mc, residual_block_with_mc


def find_predicted_block(mv, x, y, prev_frame, block_size):
    # Calculate the predicted block coordinates
    pred_x = x + mv[0]
    pred_y = y + mv[1]

    # Clip the coordinates to ensure they are within bounds
    # pred_x = np.clip(pred_x, 0, prev_frame.shape[1] - block_size)
    # pred_y = np.clip(pred_y, 0, prev_frame.shape[0] - block_size)

    predicted_block = prev_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
    return predicted_block




logger = get_logger()


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
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        mv_txt_fh = stack.enter_context(open(mv_txt_file, 'rt'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            quant_dct_coff = quant_dct_coff_fh.read(frame_size*2) # quant_dct_coff are stored as int16. i.e. 2bytes
            mv_txt =  mv_txt_fh.readline()
            if not quant_dct_coff or frame_index > frames_to_process or not mv_txt:
                break  # End of file or end of frames
            logger.debug(f"Decoding frame {frame_index}/{frames_to_process}")
            quant_dct_coff_frame = np.frombuffer(quant_dct_coff, dtype=np.int16)
            quant_dct_coff_frame = quant_dct_coff_frame.reshape((height, width))

            mv = parse_mv(mv_txt)

            if False and frame_index % params.encoder_config.I_Period == 0:
                frame = IFrame()
            else:
                frame = PFrame()
                frame.quat_dct_coffs_frame_with_mc = quant_dct_coff_frame

            decoded_frame = decode_p_frame(quant_dct_coff_frame, prev_frame, mv, params.encoder_config)

            reconstructed_frame= np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape((height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)

            logger.info(f"{frame_index:2}: psnr [{round(psnr,2):6.2f}]")


            write_y_only_frame(decoded_fh, decoded_frame)

            prev_frame = decoded_frame
    print('end decoding')







# Function to upscale U and V planes
def upscale_chroma(u_plane, v_plane):
    u_444 = zoom(u_plane, 2, order=1)  # Bilinear interpolation
    v_444 = zoom(v_plane, 2, order=1)
    return u_444, v_444


# Function to read YUV420 frames
def read_yuv420(file, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # Read Y plane
    y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

    # Read U and V planes
    u_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
    v_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

    return y_plane, u_plane, v_plane

# Function to convert YUV 4:4:4 to RGB
def yuv_to_rgb(y_plane, u_plane, v_plane):
    height, width = y_plane.shape

    # Broadcast U and V planes to the same size as Y plane
    u_plane = u_plane.reshape((height, width))
    v_plane = v_plane.reshape((height, width))

    # YUV to RGB conversion matrix
    M = np.array([[1.164, 0.0, 1.596],
                  [1.164, -0.392, -0.813],
                  [1.164, 2.017, 0.0]])

    # Normalize YUV values
    y_plane = y_plane.astype(np.float32) - 16
    u_plane = u_plane.astype(np.float32) - 128
    v_plane = v_plane.astype(np.float32) - 128

    # Stack Y, U, and V into a single array (height x width x 3)
    yuv_stack = np.stack([y_plane, u_plane, v_plane], axis=-1)

    # Apply color space conversion
    rgb_stack = np.dot(yuv_stack, M.T)

    # Clip values to [0, 255] and convert to uint8
    rgb_stack = np.clip(rgb_stack, 0, 255).astype(np.uint8)

    return rgb_stack


def create_noise_mask(height, width, noise_percent):
    mask = np.zeros((height, width), dtype=np.uint8)
    num_noise_pixels = int((noise_percent / 100.0) * height * width)
    noise_indices = np.random.choice(height * width, num_noise_pixels, replace=False)
    mask.flat[noise_indices] = 1
    return mask


def generate_random_values(height, width):
    return np.random.randint(0, 256, size=(height, width), dtype=np.uint8)


def apply_mask(channel, mask, strategy='randomize', random_values=None):
    if strategy == 'turn_off':
        return np.where(mask == 1, 0, channel)
    elif strategy == 'flip_value':
        return np.where(mask == 1, 128 + (128 - channel), channel)
    elif strategy == 'randomize':
        if random_values is None:
            raise ValueError('random_values must be provided when strategy is randomize')
        return np.where(mask == 1, random_values, channel)  # Use the precomputed random values
    return channel


def pad_or_resize_channel(channel, target_height, target_width):
    """Resize or pad the channel to fit the target size."""
    current_height, current_width = channel.shape
    if current_height != target_height or current_width != target_width:
        # Resize or pad the channel to fit the grid unit size
        padded_channel = np.zeros((target_height, target_width), dtype=np.uint8)

        # Fit the channel within the padded space (centered)
        start_y = (target_height - current_height) // 2
        start_x = (target_width - current_width) // 2

        padded_channel[start_y:start_y + current_height, start_x:start_x + current_width] = channel
        return padded_channel
    return channel


def construct_grid(channels, masks, original_width, original_height, half_border, spacing, random_values):
    grid_unit_x = int(original_width * 1.0625)
    grid_unit_y = int(original_height * 1.0625)

    num_masks = len(masks)
    grid_height = (num_masks * grid_unit_y) + ((num_masks - 1) * spacing) + 2 * half_border
    grid_width = (len(channels) * grid_unit_x) + ((len(channels) - 1) * spacing) + 2 * half_border

    # print(f"Total grid dimensions: Height = {grid_height}, Width = {grid_width}")
    # Initialize the grid buffer
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    for i, (channel_name, channel_data) in enumerate(channels.items()):
        for j, mask in enumerate(masks):
            # Apply the mask to the channel
            masked_channel = apply_mask(channel_data, mask, 'randomize', random_values)

            # Resize or pad the masked channel to fit the grid unit size
            padded_channel = pad_or_resize_channel(masked_channel, grid_unit_y, grid_unit_x)

            # Calculate position in grid (shift down for each channel)
            start_x = half_border + i * (grid_unit_x + spacing)  # Shift right for each channel
            start_y = half_border + j * (grid_unit_y + spacing)  # Shift down for each mask

            # Place the padded channel into the grid
            grid[start_y:start_y + grid_unit_y, start_x:start_x + grid_unit_x] = padded_channel

    return grid





# Function to read YUV420 frames and return only Y-plane
def read_y_component(file_path, width, height, num_frames):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)  # For U and V planes, which will be skipped

    with open(file_path, 'rb') as file:
        for _ in range(num_frames):
            y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

            # Skip U and V planes
            file.read(uv_size)
            file.read(uv_size)

            yield y_plane


# Function to save Y-only frames to individual files
def save_y_frames_to_file(params : InputParameters, frames_to_extract=None):
    if params.yuv_file is None:
        guessed_yuv_file_name = FileIOHelper(params).get_yuv_file_name()
        logger.info(f"No yuv file provided, assuming yuv = [{guessed_yuv_file_name}] ")
        params.yuv_file = guessed_yuv_file_name

    input_file = params.yuv_file

    output_file= params.y_only_file
    num_frames = frames_to_extract if frames_to_extract else calculate_num_frames(params.yuv_file, params.width, params.height)
    if os.path.exists(params.y_only_file):
        logger.info(f"y only {output_file} already exists. skipping...")
        return
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        for frame_index, y_plane in enumerate(read_y_component(input_file, params.width, params.height, num_frames)):
            f_out.write(y_plane.tobytes())


def calculate_block_average(block):
    return np.round(np.mean(block)).astype(np.uint8)

def replace_with_average(blocks):
    return np.array([np.full_like(block, calculate_block_average(block)) for block in blocks])

def reconstruct_frame_from_blocks(blocks, frame_shape, block_size):
    height, width = frame_shape
    rows = height // block_size
    cols = width // block_size
    return (blocks.reshape(rows, cols, block_size, block_size)
                   .swapaxes(1, 2)
                   .reshape(height, width))


# Function to process Y-only files and split them into blocks
def process_y_frames(params : InputParameters, block_sizes):
    input_file = params.y_only_file
    width = params.width
    height = params.height
    logger.info(f"Processing file: {input_file}")
    file_io_h = FileIOHelper(params)

    file_handles = {}

    # Open output files dynamically based on block sizes
    for block_size in block_sizes:
        file_name = file_io_h.get_file_name_wo_identifier(f'{block_size}b.y')
        if os.path.exists(file_name):
            continue
        file_handles[block_size] = open(file_name, 'wb')

    if len(file_handles) < 1:
        return

    y_size = width * height

    with open(input_file, 'rb') as f_in:
        frame_index = 0
        while True:
            y_frame = f_in.read(y_size)
            if not y_frame:
                break  # End of file

            y_plane = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))

            for block_size in block_sizes:
                # Pad the frame if necessary
                padded_frame = pad_frame(y_plane, block_size)

                # Split the frame into (block_size x block_size) blocks
                blocks = split_into_blocks(padded_frame, block_size)

                # Replace each block with its average
                averaged_blocks = replace_with_average(blocks)

                # Reconstruct the frame from blocks
                reconstructed_frame = reconstruct_frame_from_blocks(averaged_blocks, padded_frame.shape, block_size)

                # Write the reconstructed frame to the respective output file
                file_handles[block_size].write(reconstructed_frame.tobytes())

                # Example logging to show how many blocks are generated
                logger.info(f"Frame {frame_index}: Processed {len(blocks)} blocks of size {block_size}x{block_size}")

            frame_index += 1

    # Close all the file handles
    for handle in file_handles.values():
        handle.close()

def calculate_psnr_ssim(original_file, averaged_file, width, height):
    psnr_values = []
    ssim_values = []

    with open(original_file, 'rb') as orig, open(averaged_file, 'rb') as avg:
        frame_index = 0
        while True:
            orig_frame = orig.read(width * height)
            avg_frame = avg.read(width * height)
            if not orig_frame or not avg_frame:
                break  # End of file

            orig_y_plane = np.frombuffer(orig_frame, dtype=np.uint8).reshape((height, width))
            avg_y_plane = np.frombuffer(avg_frame, dtype=np.uint8).reshape((height, width))

            # Calculate PSNR and SSIM
            current_psnr = psnr(orig_y_plane, avg_y_plane)
            current_ssim = ssim(orig_y_plane, avg_y_plane)

            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)

            logger.info(f"Frame {frame_index}: PSNR = {current_psnr}, SSIM = {current_ssim}")
            frame_index += 1

    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)

    logger.info(f"Average PSNR: {average_psnr}, Average SSIM: {average_ssim}")
    return average_psnr, average_ssim

# g. Plot graphs
def plot_quality_metrics(block_sizes, psnr_values, ssim_values):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(block_sizes, psnr_values, marker='o', label='PSNR')
    plt.title('PSNR vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('PSNR (dB)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(block_sizes, ssim_values, marker='o', label='SSIM', color='orange')
    plt.title('SSIM vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('SSIM')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    # plt.show()



class FileIOHelper:
    def __init__(self, params:InputParameters):

        self.y_only_file = params.y_only_file
        self.block_size = params.encoder_config.block_size
        self.search_range = params.encoder_config.search_range
        self.quantization_factor = params.encoder_config.quantization_factor
        self.frames_to_process = params.frames_to_process

        self.file_identifier = f'{self.block_size}_{self.search_range}_{self.quantization_factor}'
        self.file_prefix = os.path.splitext(self.y_only_file)[0]

        os.makedirs(os.path.dirname(self.get_file_name(suffix='')), exist_ok=True)

    def get_file_name(self, suffix):
        return f'{self.file_prefix}/{self.file_identifier}/{suffix}'

    def get_file_name_wo_identifier(self, suffix):
        return f'{self.file_prefix}/{suffix}'

    def get_y_file_name(self):
        return  f'{self.file_prefix}.y'

    def get_yuv_file_name(self):
        return  f'{self.file_prefix}.yuv'

    def get_mv_file_name(self):
        return self.get_file_name('mv.txt')

    def get_metrics_csv_file_name(self):
        return self.get_file_name('metrics.csv')

    def get_metrics_png_file_name(self):
        return self.get_file_name('metrics.png')

    def get_mc_residual_file_name(self):
        return self.get_file_name('mc_residuals.yuv')

    def get_quant_dct_coff_fh_file_name(self):
        return self.get_file_name('mc_quant_dct_coff.bin')


    def get_mc_reconstructed_file_name(self):
        return self.get_file_name('mc_reconstructed.yuv')
    def get_mc_decoded_file_name(self):
        return self.get_file_name('mc_decoded.yuv')


def write_mv_to_file(file_handle, data, new_line_per_block = False):
    # file_handle.write(f'\nFrame: {frame_idx}\n')
    new_line_char = f'\n' if new_line_per_block else ''
    for k in sorted(data.keys()):
        file_handle.write(f'{new_line_char}{k[0]},{k[1]}:{data[k][0]},{data[k][1]}|')
    file_handle.write('\n')


def write_y_only_frame(file_handle, reconstructed_frame):
    file_handle.write(reconstructed_frame.tobytes())



def plot_metrics(params: InputParameters):
    file_io = FileIOHelper(params)

    csv_file_name = file_io.get_metrics_csv_file_name()
    frame_numbers = []
    avg_mae_values = []
    psnr_values = []

    # Read the CSV file and extract Frame Index, Average MAE, and PSNR
    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            frame, mae, psnr = row
            frame_numbers.append(int(frame))  # Frame index as integer
            avg_mae_values.append(float(mae))  # MAE value as float
            psnr_values.append(float(psnr))  # PSNR value as float

    # Generate frame numbers based on the number of MAE values
    # Plotting the metrics
    plt.figure(figsize=(10, 6))

    # Plot Average MAE
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')

    # Plot PSNR
    plt.plot(frame_numbers, psnr_values, marker='x', linestyle='--', color='r', label='PSNR')

    # Adding title and labels
    plt.title(f'MAE and PSNR per Frame, i = {params.encoder_config.block_size}, r = {params.encoder_config.search_range}, qp = {params.encoder_config.quantization_factor}')
    plt.xlabel('Frame Number')
    plt.ylabel('Metric Value')

    # Adding grid, legend, and layout
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot as a PNG file (optional)
    graph_file_name = file_io.get_metrics_png_file_name()  # You might want to rename this method for clarity
    plt.savefig(graph_file_name)

    # Close the plot to avoid display issues in some environments
    plt.close()


def parse_mv(mv_str: str):
    mv_field = {}
    mv_blocks = mv_str.strip().split('|')
    for b in mv_blocks[:-1]: # ignore last element which will be empty
        kv_pairs = b.split(':')
        cords_txt = kv_pairs[0].split(',')
        mv_txt = kv_pairs[1].split(',')
        cords = (int(cords_txt[0]), int(cords_txt[1]))
        mv = [int(mv_txt[0]), int(mv_txt[1])]
        mv_field[cords] = mv
    return mv_field



#update this for running ex4
def main(params: InputParameters):
    encode(params)
    plot_metrics(params)
    decode(params)




if __name__ == "__main__":

    encoder_parameters = EncoderConfig(
        block_size = 8,
        search_range=2,
        quantization_factor=1,
        I_Period=1,
    )

    input_params = InputParameters(
        y_only_file ='data/foreman_cif.y',
        width  = 352,
        height= 288,
        encoder_config= encoder_parameters,
        frames_to_process = 12
    )

    # ex2.main(input_params)
    # generate_sample_file(input_file, num_frames=300)

    # ex2.save_y_frames_to_file(input_params)

    # ex3.main(input_params)

    ex4.main(input_params)

