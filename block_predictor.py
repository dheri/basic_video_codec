from common import mae

def predict_block(curr_block, prev_partial_frame, block_size):
    return find_lowest_mae_block(curr_block, prev_partial_frame, block_size)

def find_lowest_mae_block(curr_block, prev_partial_frame, block_size):
    """Find the block with the lowest MAE from a smaller previous partial frame."""
    height, width = prev_partial_frame.shape
    min_mae = float('inf')
    best_mv = [0,0]  # motion vector wrt origin of prev_partial_frame

    # Loop through all possible positions in the previous partial frame

    for ref_y in range(0, height - block_size + 1):
        for ref_x in range(0, width - block_size + 1):
            ref_block = prev_partial_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            error = mae(curr_block, ref_block)

            # Update best match if a lower MAE is found, breaking ties as described
            if error < min_mae or (error == min_mae and abs(ref_x) + abs(ref_y) < abs(best_mv[0]) + abs(best_mv[1])):
                min_mae = error
                best_mv = [ref_x , ref_y]  # (dx, dy)



    return best_mv, min_mae, ref_block
