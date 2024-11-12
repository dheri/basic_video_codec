import logging
import os

import numpy as np



def get_logger():
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)-3d] %(message)s',
                        datefmt='%H:%M:%S', )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

def calculate_num_frames(file_path, width, height):
    file_size = os.path.getsize(file_path)
    frame_size = width * height + 2 * (width // 2) * (height // 2)
    return file_size // frame_size


def pad_frame(frame, block_size, pad_value=128):
    height, width = frame.shape
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        logger.warning(f"frame is padded [{pad_height} , {pad_height}]")
        padded_frame = np.full((height + pad_height, width + pad_width), pad_value, dtype=np.uint8)
        padded_frame[:height, :width] = frame
        return padded_frame
    return frame


# Function to split the frame into blocks of size (block_size x block_size)
def split_into_blocks_ex2(frame, block_size):
    height, width = frame.shape
    return (frame.reshape(height // block_size, block_size, -1, block_size)
            .swapaxes(1, 2)
            .reshape(-1, block_size, block_size))


def mae(block1, block2):
    """Compute Mean Absolute Error between two blocks."""
    return np.mean(np.abs(block1 - block2))


def generate_residual_block(curr_block, prev_frame, motion_vector, x, y, block_size):
    predicted_block_with_mc = find_mv_predicted_block(motion_vector, x, y, prev_frame, block_size).astype(np.int16)
    residual_block_with_mc = np.subtract(curr_block.astype(np.int16), predicted_block_with_mc.astype(np.int16))
    return predicted_block_with_mc, residual_block_with_mc


def find_mv_predicted_block(mv, x, y, prev_frame, block_size):
    # Calculate the predicted block coordinates
    pred_x = x + mv[0]
    pred_y = y + mv[1]


    predicted_block = prev_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
    return predicted_block


def split_into_blocks(nd_array, block_size):
    """Split the 2D array into a list of blocks."""
    height, width = nd_array.shape
    blocks = []
    # Iterate over the 2D array in steps of block_size
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = nd_array[y:y + block_size, x:x + block_size]
            blocks.append(block)
    return blocks


def merge_blocks(blocks, block_size, frame_shape):
    """
    Merges a list of blocks into a single frame matrix.

    Parameters:
    - blocks: List of 2D arrays (each block is block_size x block_size).
    - block_size: The size of each block (e.g., 4 for a 4x4 block).
    - frame_size: The size of the full frame (e.g., 16 for a 16x16 frame).

    Returns:
    - A 2D NumPy array representing the full frame.
    """
    # Calculate the number of blocks per row and column
    num_blocks_per_row = frame_shape[1] // block_size

    # Initialize an empty matrix for the full frame
    frame = np.zeros(shape=frame_shape, dtype=int)

    # Place each block in its correct position in the frame
    for idx, block in enumerate(blocks):
        # Calculate the position of the block in the full frame
        row_block = idx // num_blocks_per_row
        col_block = idx % num_blocks_per_row

        # Calculate the top-left corner coordinates for this block in the frame
        row_start = row_block * block_size
        col_start = col_block * block_size

        # Place the block into the frame
        frame[row_start:row_start + block_size, col_start:col_start + block_size] = block

    return frame


def signed_to_unsigned(value, bits):
    """Convert a signed integer to an unsigned integer."""
    if value < 0:
        return (1 << bits) + value  # Add 2^bits to negative values
    return value


def unsigned_to_signed(value, bits):
    """Convert an unsigned integer to a signed integer."""
    if value >= (1 << (bits - 1)):
        return value - (1 << bits)  # Subtract 2^bits if value is in the upper half
    return value


def int_to_3_bytes(value):
    """
    Converts an integer into a 3-byte representation.
    Assumes value is within the 24-bit range (0 to 16,777,215).
    """
    byte1 = (value >> 16) & 0xFF  # Highest 8 bits
    byte2 = (value >> 8) & 0xFF  # Middle 8 bits
    byte3 = value & 0xFF  # Lowest 8 bits
    return bytes([byte1, byte2, byte3])


def bytes_to_int_3(three_bytes):
    """
    Converts a 3-byte representation back into an integer.
    Assumes three_bytes is a bytes-like object of length 3.
    """
    return (three_bytes[0] << 16) | (three_bytes[1] << 8) | three_bytes[2]


def pad_with_zeros(array, desired_length):
    """
    Pads the array with zeros until it reaches the desired length.

    Parameters:
    - array (list): The original array to pad.
    - desired_length (int): The target length of the array.

    Returns:
    - list: Input array padded with zeros.
    """
    if len(array) < desired_length:
        zeros_to_add = desired_length - len(array)
        array.extend([0] * zeros_to_add)
    return array
