import numpy as np

from common import get_logger
from encoder.PredictionMode import PredictionMode
logger = get_logger()

class BitStreamBuffer:
    def __init__(self):
        self.byte_stream : bytearray = bytearray()  # To hold the bitstream in bytes
        self.current_byte = 0  # To accumulate bits into a byte
        self.bit_position = 0  # Current bit position in the current_byte

    def write_bit(self, bit):
        """Write a single bit to the buffer."""
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.current_byte = (self.current_byte << 1) | bit
        self.bit_position += 1

        # If the current byte is full (8 bits), append it to the stream
        if self.bit_position == 8:
            self.byte_stream.append(self.current_byte)
            self.current_byte = 0  # Reset for the next byte
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

    def write_bytes(self, byte_data):
        if not isinstance(byte_data, (bytes, bytearray)):
            raise TypeError("byte_data must be of type bytes or bytearray.")

        if self.bit_position == 0:
            # Append directly if byte-aligned
            self.buffer.extend(byte_data)
        else:
            # If not byte-aligned, append each byte bit by bit
            for byte in byte_data:
                for i in range(8):
                    self.write_bit((byte >> (7 - i)) & 1)

    def write_quantized_coeffs(self, coeffs_2d):
        """Write a 2D array of quantized coefficients to the buffer."""
        flat_coeffs = coeffs_2d.flatten()
        for coeff in flat_coeffs:
            self.write_int16(coeff)

    def read_bit(self):
        """Read the next bit from the buffer."""
        # Check if there are bits left to read
        if self.bit_position == 0:
            if len(self.byte_stream) == 0:
                raise EOFError("No more bits to read")
            # Load the next byte
            self.current_byte = self.byte_stream.pop(0)  # Consume the byte
            self.bit_position = 8  # Reset to read bits

        # Read the next bit
        self.bit_position -= 1
        return (self.current_byte >> self.bit_position) & 1
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
    def write_int8(self, value):
        """Write a signed 8-bit integer to the buffer."""
        if not (-128 <= value <= 127):
            raise OverflowError("Value out of range for int8.")
        self.byte_stream.append(value & 0xFF)

    def read_int8(self):
        """Read a signed 8-bit integer from the buffer."""
        value = int.from_bytes(self.byte_stream[:1], byteorder='big', signed=True)
        self.byte_stream = self.byte_stream[1:]  # Remove the read byte
        return value

    def write_prediction_data(self, prediction_mode, differential_data):
        if prediction_mode == PredictionMode.INTER_FRAME:
            for i in range(0, len(differential_data), 2):
                mv_x = differential_data[i]
                mv_y = differential_data[i + 1]
                self.write_int8(mv_x)  # Write mv_x
                self.write_int8(mv_y)  # Write mv_y
        elif prediction_mode == PredictionMode.INTRA_FRAME:
            for mode in differential_data:
                self.write_bit(mode)
        else:
            raise ValueError("Invalid prediction mode")

    def read_prediction_data(self, prediction_mode, num_blocks):
        prediction_data = []
        if prediction_mode == PredictionMode.INTER_FRAME:
            for _ in range(num_blocks):
                mv_x = self.read_int8()
                mv_y = self.read_int8()
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

    def read_quantized_coeffs(self, width, height):
        """Read a specified number of 16-bit quantized coefficients from the buffer."""
        coeffs = []
        for _ in range(width * height):
            coeffs.append(self.read_int16())
        return np.array(coeffs).reshape(int(height), int(width))

    def flush(self):
        """Flush remaining bits in the buffer to the byte stream."""
        if self.bit_position > 0:
            # Pad the remaining bits with zeros to complete the byte
            self.current_byte <<= (8 - self.bit_position)
            self.byte_stream.append(self.current_byte)
            self.current_byte = 0  # Reset for the next byte
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

