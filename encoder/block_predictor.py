import math
from collections import deque

from common import mae, logger
from encoder.params import EncoderConfig



def find_fast_me_block(curr_block, curr_block_cords, mvp, reference_frame, ec:EncoderConfig, comparison_count):
    origin = curr_block_cords
    candidates = {
        "origin": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, 0, 0, ec.block_size)),
        "pmv": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1], ec.block_size)),
        "pmv_top": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1] - 1, ec.block_size)),
        "pmv_right": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0] + 1, mvp[1], ec.block_size)),
        "pmv_bottom": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1] + 1, ec.block_size)),
        "pmv_left": lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0] - 1, mvp[1], ec.block_size)),
    }
    mv_map = {
        "origin": (0, 0),
        "pmv": (mvp[0], mvp[1]),
        "pmv_top": (mvp[0], mvp[1] - 1),
        "pmv_right": (mvp[0] + 1, mvp[1]),
        "pmv_bottom": (mvp[0], mvp[1] + 1),
        "pmv_left": (mvp[0] - 1, mvp[1]),
    }
    min_mae = math.inf
    best_key = None

    for key, func in candidates.items():
        try:
            current_mae = func()
            comparison_count += 1
            if current_mae < min_mae:
                min_mae = current_mae
                best_key = key
        except Exception:
            # Ignore errors from invalid positions
            continue

    # If the best match is "origin", return its motion vector
    if best_key == "origin" or best_key == "pmv":
        return mv_map[best_key], min_mae, candidates[best_key], comparison_count

    # Otherwise, update the origin to the best MV and recurse
    best_mv = mv_map[best_key]
    # logger.debug(f" {best_key} ")
    # new_origin = (origin[0] + best_mv[0], origin[1] + best_mv[1])
    return find_fast_me_block(curr_block, origin, best_mv, reference_frame, ec, comparison_count)


def find_lowest_mae_block(curr_block, curr_block_cords, reference_frames: deque, block_size, search_range):
    origin = curr_block_cords
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
                if is_out_of_range(mv_x, mv_y, origin, block_size, width, height):
                    continue
                counter += 1
                ref_block = reference_frame[
                            origin[1]+ mv_y : origin[1] + mv_y + block_size,
                            origin[0]+ mv_x : origin[0] + mv_x + block_size
                            ]

                error = mae(curr_block, ref_block)

                # Update best match if a lower MAE is found, breaking ties as described
                if error < min_mae or (error == min_mae and abs(mv_x) + abs(mv_y) < abs(best_mv[0]) + abs(best_mv[1])):
                    min_mae = error
                    best_mv = [mv_x, mv_y, ref_frame_idx]  # (dx, dy)
    return best_mv, min_mae, ref_block, (len(reference_frames) * ((2 * search_range + 1) ** 2 ))

def get_ref_block_at_mv(reference_frame, origin, mv_x, mv_y, block_size):
    if is_out_of_range(mv_x, mv_y, origin, block_size, reference_frame.shape[1], reference_frame.shape[0]):
        raise ValueError(f"Out of range")
    ref_block = reference_frame[
                origin[1] + mv_y: origin[1] + mv_y + block_size,
                origin[0] + mv_x: origin[0] + mv_x + block_size
                ]
    return ref_block

def is_out_of_range(mv_x, mv_y, origin, block_size, width, height):
    return origin[0] + mv_x < 0 or origin[1] + mv_y < 0 or origin[0] + mv_x + block_size > width or origin[1] + mv_y + block_size > height
