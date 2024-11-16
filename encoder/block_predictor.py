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


def predict_block(curr_block, prev_partial_frames:list, block_size):
    return find_lowest_mae_block(curr_block, prev_partial_frames, block_size)


def find_lowest_mae_block(curr_block, prev_partial_frames: list, block_size):
    """Find the block with the lowest MAE from a smaller previous partial frame."""
    height, width = prev_partial_frames[0].shape
    if width < block_size or height < block_size:
        raise ValueError(f"width [{width}] or height [{height}] of given block  < block_size [{block_size}]")
    min_mae = float('inf')
    best_mv = [0, 0, 0]  # motion vector wrt origin of prev_partial_frame

    # Loop through all possible positions in the previous partial frame
    ref_block = None
    for ref_frame_idx, prev_partial_frame  in enumerate(prev_partial_frames):
        for ref_y in range(0, height - block_size + 1):
            for ref_x in range(0, width - block_size + 1):
                ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                error = mae(curr_block, ref_block)

                # Update best match if a lower MAE is found, breaking ties as described
                if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                    min_mae = error
                    best_mv = [ref_x, ref_y, ref_frame_idx]  # (dx, dy)

    return best_mv, min_mae, ref_block


def compute_rd_cost(residual_block, quantized_dct_coffs, block_size, lambda_value):
    """
    Compute RD cost for a given block.
    RD cost = Distortion + lambda * Rate
    """
    # Distortion using Sum of Absolute Differences (SAD)
    distortion = np.sum(np.abs(residual_block))
    # Rate approximation using quantized coefficients
    rate = np.sum(np.abs(quantized_dct_coffs) > 0)  # Count of non-zero coefficients
    # RD cost
    rd_cost = distortion + lambda_value * rate
    return rd_cost

def process_block_vbs(curr_block, reconstructed_frame, x, y, block_size, quantization_factor, lambda_value, dct_fn):
    """
    Process a block and decide whether to split into smaller blocks or encode as a larger block.
    """
    # Encode as a single block
    predicted_block, mode, mae = intra_predict_block(curr_block, reconstructed_frame, x, y, block_size)
    residual_block = curr_block - predicted_block
    quantized_dct_coffs = dct_fn(residual_block, block_size, quantization_factor)
    rd_cost_large = compute_rd_cost(residual_block, quantized_dct_coffs, block_size, lambda_value)

    # Split into 4 sub-blocks and calculate RD cost
    sub_block_size = block_size // 2
    sub_blocks = []
    rd_cost_small = 0
    for sub_y in range(2):
        for sub_x in range(2):
            sub_x_start = x + sub_x * sub_block_size
            sub_y_start = y + sub_y * sub_block_size
            curr_sub_block = curr_block[sub_y_start - y:sub_y_start - y + sub_block_size,
                                        sub_x_start - x:sub_x_start - x + sub_block_size]
            predicted_sub_block, sub_mode, sub_mae = intra_predict_block(
                curr_sub_block, reconstructed_frame, sub_x_start, sub_y_start, sub_block_size)
            residual_sub_block = curr_sub_block - predicted_sub_block
            quantized_sub_dct_coffs = dct_fn(residual_sub_block, sub_block_size, quantization_factor)
            rd_cost_small += compute_rd_cost(residual_sub_block, quantized_sub_dct_coffs, sub_block_size, lambda_value)
            sub_blocks.append({
                "coords": (sub_x_start, sub_y_start),
                "mode": sub_mode,
                "mae": sub_mae,
                "quantized_dct_coffs": quantized_sub_dct_coffs,
                "residual": residual_sub_block,
                "predicted_block": predicted_sub_block
            })

    # Decide based on RD costs
    if rd_cost_large <= rd_cost_small:
        return {
            "split": False,
            "mode": mode,
            "mae": mae,
            "quantized_dct_coffs": quantized_dct_coffs,
            "residual": residual_block,
            "predicted_block": predicted_block
        }
    else:
        return {"split": True, "sub_blocks": sub_blocks}


