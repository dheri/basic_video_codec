import math
from collections import deque

from common import mae, logger
from encoder.params import EncoderConfig



def find_fast_me_block(curr_block, curr_block_cords, mvp, reference_frames, ec:EncoderConfig, comparison_count):
    origin = curr_block_cords
    min_mae = math.inf
    best_mv_key = None
    mv_map = {"origin0" : (0,0,0)}
    candidates = {}

    for rf_idx, reference_frame  in enumerate(reference_frames):

        candidates[f"origin{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, 0, 0, ec.block_size))
        candidates[f"pmv{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1], ec.block_size))
        candidates[f"pmv_top{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1] - 1,  ec.block_size))
        candidates[f"pmv_right{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin,  mvp[0] + 1, mvp[1], ec.block_size))
        candidates[f"pmv_bottom{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0], mvp[1] + 1, ec.block_size))
        candidates[f"pmv_left{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(reference_frame, origin, mvp[0] - 1, mvp[1], ec.block_size))

        mv_map[f"origin{rf_idx}"] = (0, 0, rf_idx)
        mv_map[f"pmv{rf_idx}"] = (mvp[0], mvp[1], rf_idx)
        mv_map[f"pmv_top{rf_idx}"] = (mvp[0], mvp[1] - 1, rf_idx)
        mv_map[f"pmv_right{rf_idx}"] = (mvp[0] + 1, mvp[1], rf_idx)
        mv_map[f"pmv_bottom{rf_idx}"] = (mvp[0], mvp[1] + 1, rf_idx)
        mv_map[f"pmv_left{rf_idx}"] = (mvp[0] - 1, mvp[1], rf_idx)

        mv_map[f"origin{rf_idx}"] = (0, 0, rf_idx)
        mv_map[f"pmv{rf_idx}"] = (mvp[0], mvp[1], rf_idx)
        mv_map[f"pmv_top{rf_idx}"] = (mvp[0], mvp[1] - 1, rf_idx)
        mv_map[f"pmv_right{rf_idx}"] = (mvp[0] + 1, mvp[1], rf_idx)
        mv_map[f"pmv_bottom{rf_idx}"] = (mvp[0], mvp[1] + 1, rf_idx)
        mv_map[f"pmv_left{rf_idx}"] = (mvp[0] - 1, mvp[1], rf_idx)

        for key, func in candidates.items():
            try:
                current_mae = func()
                comparison_count += 1
                if current_mae < min_mae:
                    min_mae = current_mae
                    best_mv_key = key
            except Exception as e:
                # Ignore errors from invalid positions
                # logger.error(e)
                continue

    # If the best match is "origin", return its motion vector
    if "origin" in best_mv_key or   "pmv" in best_mv_key:
        return mv_map[best_mv_key], min_mae, candidates[best_mv_key], comparison_count

    # Otherwise, update the origin to the best MV and recurse
    best_mv = mv_map[best_mv_key]
    # logger.debug(f" {best_mv_key} ")
    # new_origin = (origin[0] + best_mv[0], origin[1] + best_mv[1])
    return find_fast_me_block(curr_block, origin, best_mv, reference_frames, ec, comparison_count)


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
