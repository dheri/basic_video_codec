from collections import deque

import numpy as np

from common import mae, logger


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


def predict_block(curr_block, curr_block_cords, reference_frames:deque, block_size, search_range):
    return find_lowest_mae_block(curr_block, curr_block_cords, reference_frames, block_size, search_range)


def find_lowest_mae_block(curr_block, curr_block_cords, reference_frames: deque, block_size, search_range):
    orign = curr_block_cords
    height, width = reference_frames[0].shape
    if width < block_size or height < block_size:
        raise ValueError(f"width [{width}] or height [{height}] of given block  < block_size [{block_size}]")
    min_mae = float('inf')
    best_mv = [0, 0, 0]
    ref_block = None
    counter = 0
    for ref_frame_idx, reference_frame  in enumerate(reference_frames):
        for mv_y in range(-search_range, search_range+1):
            for mv_x in range(-search_range, search_range+1):
                if is_out_of_range(mv_x, mv_y, orign, block_size, width, height):
                    continue
                counter += 1
                ref_block = reference_frame[
                            orign[1]+ mv_y : orign[1] + mv_y + block_size,
                            orign[0]+ mv_x : orign[0] + mv_x + block_size
                            ]

                error = mae(curr_block, ref_block)

                # Update best match if a lower MAE is found, breaking ties as described
                if error < min_mae or (error == min_mae and abs(mv_x) + abs(mv_y) < abs(best_mv[0]) + abs(best_mv[1])):
                    min_mae = error
                    best_mv = [mv_x, mv_y, ref_frame_idx]  # (dx, dy)
    return best_mv, min_mae, ref_block

def is_out_of_range(mv_x, mv_y, orign, block_size, width, height):
    return orign[0] + mv_x < 0 or orign[1] + mv_y < 0 or orign[0] + mv_x + block_size > width or orign[1] + mv_y + block_size > height
