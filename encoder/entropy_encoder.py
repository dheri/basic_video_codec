import math

import numpy as np
from bitarray import bitarray

from common import get_logger

logger = get_logger()


def exp_golomb_encode(value):
    """
    Encodes an integer using Exp-Golomb encoding.
    """
    if value == 0:
        return '0'

    # Map value for Exp-Golomb: positive values -> 2*value - 1, negative values -> -2*value
    mapped_value = 2 * value - 1 if value > 0 else -2 * value

    # Calculate prefix length (number of leading zeros)
    m = int(math.log2(mapped_value + 1))
    prefix = '0' * m + '1'

    # Calculate suffix
    suffix = format(mapped_value - (1 << m), f'0{m}b')

    return prefix + suffix


def exp_golomb_decode(bitstream):
    """
    Decodes an Exp-Golomb encoded integer from a bitstream.
    This optimized version operates directly on the bitarray without string conversions.
    """
    # Find the number of leading zeros (m), directly in the bitarray
    m = 0
    while m < len(bitstream) and not bitstream[m]:
        m += 1

    logger.info(f"m [{m:3}] bit_str: {bitstream[m + 1:m + 1 + m]}")

    # Check if we reached the end of the bitstream without finding a '1'
    if m >= len(bitstream):
        raise ValueError("Not enough bits to decode the exp-Golomb code (prefix error).")

    # Calculate the value of the Exp-Golomb code
    # The prefix '1' bit is at position m, followed by m bits for the suffix
    value = (1 << m)  # This is 2^m, the base value

    # Make sure there are enough bits to read the suffix
    if m > 0:
        if m + 1 + m > len(bitstream):  # Ensure there are m bits after the '1' prefix
            raise ValueError("Not enough bits in the stream to decode the suffix.")

        # Extract the suffix using bit shifts
        suffix = 0
        for i in range(m):
            suffix = (suffix << 1) | bitstream[m + 1 + i]
        value += suffix

    # Map back to the original signed integer
    decoded_value = (value + 1) // 2 if value % 2 else -(value // 2)

    # Create the remaining bitstream by slicing from the next position after the suffix
    remaining_bitstream = bitstream[m + 1 + m:]
    return decoded_value, remaining_bitstream


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
