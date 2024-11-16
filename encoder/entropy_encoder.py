import numpy as np
from bitarray import bitarray

def exp_golomb_encode(value):
    if value == 0:
        return '0'
    if value > 0:
        mapped_value = 2 * value - 1
    else:
        mapped_value = -2 * value
    m = int(np.log2(mapped_value + 1))
    prefix = '0' * m + '1'
    suffix = format(mapped_value - (1 << m), f'0{m}b')
    return prefix + suffix

def exp_golomb_decode(bitstream):
    # Ensure we are working with a bitarray and convert it to string for easier parsing
    bit_str = bitstream.to01()  # Convert bitstream to binary string

    m = 0
    # Find the first '1' in the bitstream (this is the start of the Golomb code)
    while m < len(bit_str) and bit_str[m] == '0':
        m += 1

    if m >= len(bit_str):  # If we run out of bits, return an error
        raise ValueError("Not enough bits to decode the exp-Golomb code.")

    # Compute the value from the Golomb code (m leading zeroes followed by '1' and m-bit suffix)
    if m + 1 + m > len(bit_str):  # Ensure we have enough bits for suffix
        raise ValueError("Insufficient bits in the stream to decode.")

    value = (1 << m) + int(bit_str[m + 1:m + 1 + m], 2)

    # Return the decoded symbol and the remaining bitstream
    remaining_bitstream = bitarray(bit_str[m + 1 + m:])
    return value, remaining_bitstream

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


def exp_golomb_encode(value):
    if value == 0:
        return '0'
    if value > 0:
        mapped_value = 2 * value - 1
    else:
        mapped_value = -2 * value
    m = int(np.log2(mapped_value + 1))
    prefix = '0' * m + '1'
    suffix = format(mapped_value - (1 << m), f'0{m}b')
    return prefix + suffix

def encode_split_flag(bitstream, is_split):
    bitstream.append(1 if is_split else 0)

def encode_sub_blocks(bitstream, sub_blocks):
    for sub_block in sub_blocks:
        for value in sub_block.flatten():
            bitstream.extend(exp_golomb_encode(value))
