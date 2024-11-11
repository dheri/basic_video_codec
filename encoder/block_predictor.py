import numpy as np

from common import mae


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
    best_mv = [0, 0]  # motion vector wrt origin of prev_partial_frame

    # Loop through all possible positions in the previous partial frame
    ref_block = None

    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            # Update best match if a lower MAE is found, breaking ties as described
            if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = [ref_x, ref_y]  # (dx, dy)

    return best_mv, min_mae, ref_block
