from collections import deque

from common import mae


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
