from bitarray.util import ba2hex

from common import get_logger

logger = get_logger()

import numpy as np
from bitarray import bitarray
from common import get_logger, split_into_blocks
from encoder.PredictionMode import PredictionMode
from input_parameters import InputParameters

logger = get_logger()


class BitStreamBuffer:
    def __init__(self):
        self.bit_stream = bitarray()  # Use bitarray to hold the bitstream
        self.bit_position = 0  # Current bit position in the bitarray

    def write_bit(self, bit):
        """Write a single bit to the buffer."""
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.bit_stream.append(bit)

    def write_bits(self, value, num_bits):
        """Write a value as a series of bits."""
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.write_bit(bit)

    def write_int16(self, value):
        """Write a 16-bit signed integer to the bitstream."""
        if not (-32768 <= value <= 32767):
            raise OverflowError("Value out of range for int16.")
        # Write high byte and low byte (big-endian)
        self.write_bits((value >> 8) & 0xFF, 8)  # High byte
        self.write_bits(value & 0xFF, 8)  # Low byte

    def write_int8(self, value):
        """Write a signed 8-bit integer to the buffer."""
        if not (-128 <= value <= 127):
            raise OverflowError("Value out of range for int8.")
        unsigned_byte = (value + 256) % 256
        self.write_bits(unsigned_byte, 8)

    def read_bit(self):
        """Read the next bit from the buffer."""
        if self.bit_position >= len(self.bit_stream):
            raise EOFError("No more bits to read")

        bit = self.bit_stream[self.bit_position]
        self.bit_position += 1
        return bit

    def read_bits(self, num_bits):
        """Read a specified number of bits from the buffer."""
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        # logger.info(f"read {num_bits} bits : [ u {value:3d} , s {unsigned_to_signed(value, num_bits):4d}] ")
        return value

    def read_int16(self):
        """Read a 16-bit signed integer from the buffer."""
        value = self.read_bits(16)  # Read 16 bits
        if value >= 0x8000:
            value -= 0x10000
        return value

    def read_int8(self):
        """Read a signed 8-bit integer from the buffer."""
        value = self.read_bits(8)
        signed_value = value if value < 128 else value - 256
        return signed_value

    def get_bitstream(self):
        """Return the byte stream as bytes."""
        return self.bit_stream.tobytes()

    def flush(self):
        # raise ValueError("flush not needed anymore")
        missing_bits = len(self.bit_stream) % 8
        for i in range(8 - missing_bits):
            self.bit_stream.append(0)

    def write_quantized_coeffs(self, quantized_dct_residual_frame, block_size):
        """Write a 2D array of quantized coefficients to the buffer block-wise."""
        # Split the coefficients into blocks
        blocks = split_into_blocks(quantized_dct_residual_frame, block_size)

        for block in blocks:
            for coeff in block.flatten():  # Flatten the block to write coefficients sequentially
                self.write_int16(coeff)

    def write_prediction_data(self, prediction_mode, differential_data):
        if prediction_mode == PredictionMode.INTER_FRAME:
            for i in range(0, len(differential_data), 2):
                mv_x = differential_data[i]
                mv_y = differential_data[i + 1]
                # differential_data  will be unit 8 it
                self.write_bits(mv_x, 8)
                self.write_bits(mv_y, 8)
        elif prediction_mode == PredictionMode.INTRA_FRAME:
            for mode in differential_data:
                self.write_bit(mode)
        else:
            raise ValueError("Invalid prediction mode")

    def read_prediction_data(self, prediction_mode, params: InputParameters):
        num_blocks = (params.height // params.encoder_config.block_size) * (
                    params.width // params.encoder_config.block_size)

        prediction_data = bytearray()
        if prediction_mode == PredictionMode.INTER_FRAME:
            for _ in range(num_blocks):
                mv_x = self.read_bits(8)
                mv_y = self.read_bits(8)
                prediction_data.append(mv_x)
                prediction_data.append(mv_y)
        elif prediction_mode == PredictionMode.INTRA_FRAME:
            # Read prediction modes (0 for horizontal, 1 for vertical)
            for _ in range(num_blocks):
                mode = self.read_bit()
                prediction_data.append(mode)
        else:
            raise ValueError("Invalid prediction mode")

        return prediction_data

    # def read_quantized_coeffs(self, width, height):
    #     """Read a specified number of 16-bit quantized coefficients from the buffer."""
    #     coeffs = []
    #     for _ in range(width * height):
    #         coeffs.append(self.read_int16())
    #     return np.array(coeffs).reshape(int(height), int(width))

    def read_quantized_coeffs(self, height, width, block_size):
        coeffs_frame = np.empty(shape=(height, width))

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                coff_block = np.empty(block_size ** 2)
                for i in range(block_size ** 2):
                    coff_block[i] = self.read_int16()

                coeffs_frame[y:y + block_size, x:x + block_size] = coff_block.reshape(block_size, block_size)

        return coeffs_frame

    def __repr__(self):
        bin_rep = ''.join(f'{byte:08b}' for byte in bytes(self.bit_stream))
        return f"hex:\t{ba2hex(self.bit_stream)} \nbin:\t{bin_rep}"


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
