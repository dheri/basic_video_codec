import math

import numpy as np
from bitarray import bitarray

from common import get_logger

logger = get_logger()



def exp_golomb_encode(value):
    """
    Encodes a non-negative integer using Exp-Golomb encoding.
    """
    mapped_value  =  -2 * value if value <= 0 else 2 * value - 1

    # Step 1: Calculate (mapped_value + 1) in binary
    encoded_value = mapped_value + 1

    # Step 2: Calculate the binary representation of encoded_value
    binary_rep = bitarray()
    binary_rep.frombytes(encoded_value.to_bytes((encoded_value.bit_length() + 7) // 8, byteorder='big'))

    # Remove leading zero bits that may result from byte padding
    binary_rep = binary_rep[binary_rep.index(1):]

    # Step 3: Count the bits in binary representation and add leading zeros
    num_bits = len(binary_rep)
    leading_zeros = bitarray('0' * (num_bits - 1))
    result = leading_zeros + binary_rep  # Concatenate the leading zeros with the binary representation

    return result


def exp_golomb_decode(bitstream):
    """
    Decodes an Exp-Golomb encoded integer from a bitstream.
    This optimized version operates directly on the bitarray without string conversions.
    """
    # Find the number of leading zeros (m), directly in the bitarray
    m = 0
    while m < len(bitstream) and not bitstream[m]:
        m += 1

    # if len(bitstream) < 12:
    #     logger.info(f"m [{m:3}] bit_str: {bitstream[m + 1:m + 1 + m]}, remaining len:{len(bitstream):6d}")
    #     logger.info(f"{bitstream}")

    # Check if we reached the end of the bitstream without finding a '1'
    if m >= len(bitstream):
        # raise ValueError("Not enough bits to decode the exp-Golomb code (prefix error).")
        logger.info(f" leftover bitstream: {bitstream} ")

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
            # Only add the zero count if there are remaining non-zero terms
            if i < len(coeffs):
                encoded.append(zero_count)  # Positive for run of zeros
            else:
                encoded.append(0)  # Indicates end of meaningful data (all remaining zeros)
        else:
            nonzero_count = 0
            start_idx = i
            while i < len(coeffs) and coeffs[i] != 0:
                nonzero_count += 1
                i += 1
            encoded.append(-nonzero_count)  # Negative for run of non-zeros
            encoded.extend(coeffs[start_idx:i])  # Append the actual non-zero terms

    return encoded


def rle_decode(encoded):
    decoded = []
    i = 0
    while i < len(encoded):
        count = encoded[i]

        if count == 0:
            # 0 indicates that the rest of the values are all zeros, stop decoding
            break
        elif count > 0:
            # Positive count represents a run of zeros
            decoded.extend([0] * count)
        else:
            # Negative count represents non-zero terms, append the following terms
            count = -count  # Convert to positive to get the number of terms
            i += 1
            decoded.extend(encoded[i:i + count])
            i += count - 1  # Move index past the non-zero terms

        i += 1  # Move to the next element in encoded list

    return decoded



def zigzag_order(matrix):
    """
    Takes a square matrix and returns a list of elements in zigzag order.
    """
    n = len(matrix)  # Assuming a square matrix
    result = []

    # Traverse diagonals
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # For even diagonals, go from bottom-left to top-right
            for i in range(s + 1):
                if i < n and (s - i) < n:
                    result.append(matrix[i][s - i])
        else:
            # For odd diagonals, go from top-right to bottom-left
            for i in range(s + 1):
                if i < n and (s - i) < n:
                    result.append(matrix[s - i][i])

    return result


def inverse_zigzag_order(arr, n):
    """
    Takes a zigzag-ordered list and reconstructs the original n x n square matrix.
    """
    matrix = [[0] * n for _ in range(n)]
    idx = 0

    # Traverse diagonals
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # For even diagonals, go from bottom-left to top-right
            for i in range(s + 1):
                if i < n and (s - i) < n:
                    matrix[i][s - i] = arr[idx]
                    idx += 1
        else:
            # For odd diagonals, go from top-right to bottom-left
            for i in range(s + 1):
                if i < n and (s - i) < n:
                    matrix[s - i][i] = arr[idx]
                    idx += 1

    return matrix

