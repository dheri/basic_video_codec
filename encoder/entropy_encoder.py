"""
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


"""


import numpy as np
from bitarray import bitarray

def exp_golomb_encode(value):
    if value == 0:
        return bitarray('0')
    if value > 0:
        mapped_value = 2 * value - 1
    else:
        mapped_value = -2 * value
    m = int(np.log2(mapped_value + 1))
    prefix = bitarray('0' * m + '1')
    suffix = bitarray(format(mapped_value - (1 << m), f'0{m}b'))
    return prefix + suffix

def exp_golomb_decode(bitstream):
    m = 0
    while m < len(bitstream) and not bitstream[m]:
        m += 1

    if m >= len(bitstream):
        raise ValueError("Not enough bits to decode the exp-Golomb code.")

    value = (1 << m) + int(bitstream[m + 1:m + 1 + m].to01(), 2)
    remaining_bitstream = bitstream[m + 1 + m:]
    return value, remaining_bitstream

def encode_split_flag(bitstream, is_split):
    """Encodes whether a block is split."""
    bitstream.append(1 if is_split else 0)

def decode_split_flag(bitstream):
    """Decodes whether a block is split."""
    is_split = bitstream.pop(0)
    return bool(is_split), bitstream

def encode_sub_blocks(bitstream, sub_blocks):
    """Encodes sub-blocks using exp-Golomb coding."""
    for sub_block in sub_blocks:
        for value in sub_block.flatten():
            bitstream.extend(exp_golomb_encode(value))

def decode_sub_blocks(bitstream, block_size, sub_block_size):
    """Decodes sub-block data."""
    sub_blocks = []
    for _ in range((block_size // sub_block_size) ** 2):  # Number of sub-blocks
        sub_block = []
        for _ in range(sub_block_size * sub_block_size):  # Number of values in a sub-block
            value, bitstream = exp_golomb_decode(bitstream)
            sub_block.append(value)
        sub_blocks.append(np.array(sub_block).reshape(sub_block_size, sub_block_size))
    return sub_blocks, bitstream

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

def rle_decode(encoded):
    decoded = []
    i = 0
    while i < len(encoded):
        run_length = encoded[i]
        if run_length > 0:
            decoded.extend([0] * run_length)
        elif run_length < 0:
            decoded.extend(encoded[i + 1:i + 1 - run_length])
            i += -run_length
        else:
            break
        i += 1
    return decoded
import numpy as np

def zigzag_order(block):
    rows, cols = block.shape
    solution = [[] for _ in range(rows + cols - 1)]
    for i in range(rows):
        for j in range(cols):
            sum_idx = i + j
            if sum_idx % 2 == 0:
                solution[sum_idx].insert(0, block[i][j])
            else:
                solution[sum_idx].append(block[i][j])
    return [item for sublist in solution for item in sublist]

import numpy as np

def inverse_zigzag_order(zigzag_list, block_size):
    block = np.zeros((block_size, block_size), dtype=np.float32)
    index = 0
    for i in range(2 * block_size - 1):
        if i % 2 == 0:
            for j in range(min(i + 1, block_size)):
                if i < block_size:
                    block[i - j][j] = zigzag_list[index]
                else:
                    block[block_size - 1 - j][i - (block_size - 1 - j)] = zigzag_list[index]
                index += 1
        else:
            for j in range(min(i + 1, block_size)):
                if i < block_size:
                    block[j][i - j] = zigzag_list[index]
                else:
                    block[i - (block_size - 1 - j)][block_size - 1 - j] = zigzag_list[index]
                index += 1
    return block
