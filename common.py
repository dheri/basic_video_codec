import os

import numpy as np


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