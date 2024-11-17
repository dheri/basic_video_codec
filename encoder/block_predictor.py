import math
from collections import deque

import numpy as np
from scipy.ndimage import map_coordinates

from common import mae, logger
from encoder.Frame import Frame
from encoder.params import EncoderConfig



def find_fast_me_block(curr_block, curr_block_cords, mvp, pFrame:Frame, ec:EncoderConfig, comparison_count):
    reference_frames = pFrame.reference_frames
    interpolated_reference_frames = pFrame.interpolated_reference_frames
    origin = curr_block_cords
    min_mae = math.inf
    best_mv_key = None
    mv_map = {"origin0" : (0,0,0)}
    candidates = {}

    for rf_idx, rf  in enumerate(reference_frames):
        irf = interpolated_reference_frames[rf_idx]

        candidates[f"origin{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin, 0, 0, ec))
        candidates[f"pmv{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin, mvp[0], mvp[1], ec))
        candidates[f"pmv_top{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin, mvp[0], mvp[1] - 1,  ec))
        candidates[f"pmv_right{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin,  mvp[0] + 1, mvp[1], ec))
        candidates[f"pmv_bottom{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin, mvp[0], mvp[1] + 1, ec))
        candidates[f"pmv_left{rf_idx}"] = lambda: mae(curr_block, get_ref_block_at_mv(rf, irf, origin, mvp[0] - 1, mvp[1], ec))

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


def find_lowest_mae_block(curr_block, curr_block_cords, pFrame: Frame, ec:EncoderConfig):
    origin = curr_block_cords

    block_size, search_range = ec.block_size, ec.search_range
    if ec.fracMeEnabled:
        search_range *= 2

    reference_frames = pFrame.reference_frames
    height, width = reference_frames[0].shape
    if width < block_size or height < block_size:
        raise ValueError(f"width [{width}] or height [{height}] of given block  < block_size [{block_size}]")
    min_mae = float('inf')
    best_mv = [0, 0, 0]
    ref_block = None
    counter = 0
    for ref_frame_idx, reference_frame  in enumerate(reference_frames):
        irf = pFrame.interpolated_reference_frames[ref_frame_idx]
        for mv_y in range(-search_range, search_range+1):
            for mv_x in range(-search_range, search_range+1):
                curr_mv = [mv_x, mv_y, ref_frame_idx]
                try:
                    ref_block = get_ref_block_at_mv(reference_frame, irf, origin, mv_x, mv_y, ec)
                except Exception as e:
                    continue
                counter += 1
                error = mae(curr_block, ref_block)
                # Update best match if a lower MAE is found, breaking ties as described
                if error < min_mae or (error == min_mae and abs(mv_x) + abs(mv_y) < abs(best_mv[0]) + abs(best_mv[1])):
                    min_mae = error
                    best_mv = curr_mv
    return best_mv, min_mae, ref_block, (len(reference_frames) * ((2 * search_range + 1) ** 2 ))

def get_ref_block_at_mv(ref_frame, interpolated_ref_frame, origin, mv_x, mv_y, ec: EncoderConfig):
    block_size = ec.block_size
    if is_out_of_range(mv_x, mv_y, origin, ref_frame, interpolated_ref_frame, ec):
        raise ValueError(f"Motion vector ({mv_x}, {mv_y}) is out of range.")

    if not ec.fracMeEnabled:
        # Integer Motion Estimation: Direct pixel extraction
        ref_block = ref_frame[
            origin[1] + mv_y: origin[1] + mv_y + block_size,
            origin[0] + mv_x: origin[0] + mv_x + block_size
                    ]
    else:
        interp_mv_x = (origin[0]) * 2 + mv_x
        interp_mv_y = (origin[1]) * 2 + mv_y

        ref_block = interpolated_ref_frame[
            interp_mv_y : interp_mv_y + block_size * 2 : 2,  # Step by 2 to get back to original block size
            interp_mv_x : interp_mv_x + block_size * 2 : 2,
        ]


    return ref_block

def is_out_of_range(mv_x, mv_y, origin, r_frame, interpolated_ref_frame, ec):
    frame = interpolated_ref_frame if ec.fracMeEnabled else r_frame
    width, height = frame.shape[1], frame.shape[0]

    # Adjust block size and motion vector scaling for fractional ME
    if ec.fracMeEnabled:
        # Scale origin, motion vector, and block size by 2 for the interpolated frame
        origin_x_scaled = origin[0] * 2
        origin_y_scaled = origin[1] * 2
        mv_x_scaled = mv_x
        mv_y_scaled = mv_y
        block_size_scaled = ec.block_size * 2

        # Check bounds for the interpolated frame
        return (
            origin_x_scaled + mv_x_scaled < 0 or
            origin_y_scaled + mv_y_scaled < 0 or
            origin_x_scaled + mv_x_scaled + block_size_scaled > width or
            origin_y_scaled + mv_y_scaled + block_size_scaled > height
        )
    else:
        # Integer Motion Estimation: Regular bounds check
        return (
            origin[0] + mv_x < 0 or
            origin[1] + mv_y < 0 or
            origin[0] + mv_x + ec.block_size > width or
            origin[1] + mv_y + ec.block_size > height
        )

def build_pre_interpolated_buffer(reference_frame):
    reference_frame = reference_frame.astype(np.int16)
    height, width = reference_frame.shape
    pre_interpolated = np.zeros((2 * height, 2 * width), dtype=np.uint8)

    # Populate the buffer
    for y in range(height):
        for x in range(width):
            # Original pixel (integer positions)
            pre_interpolated[2 * y, 2 * x] = reference_frame[y, x]

            # Horizontal interpolation (x + 0.5)
            if x + 1 < width:
                pre_interpolated[2 * y, 2 * x + 1] = np.ceil((
                    reference_frame[y, x] + reference_frame[y, x + 1]
                ) / 2)

            # Vertical interpolation (y + 0.5)
            if y + 1 < height:
                pre_interpolated[2 * y + 1, 2 * x] = np.ceil((
                    reference_frame[y, x] + reference_frame[y + 1, x]
                ) / 2)

            # Diagonal interpolation (x + 0.5, y + 0.5)
            if x + 1 < width and y + 1 < height:
                pre_interpolated[2 * y + 1, 2 * x + 1] = np.ceil((
                    reference_frame[y, x]
                    + reference_frame[y, x + 1]
                    + reference_frame[y + 1, x]
                    + reference_frame[y + 1, x + 1]
                ) / 4)

    return pre_interpolated
