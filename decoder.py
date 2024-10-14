import logging

import numpy as np

from file_io import write_y_only_frame
from motion_vector import parse_mv

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def find_predicted_block(mv, x, y, prev_frame, block_size):
    # Calculate the predicted block coordinates
    pred_x = x + mv[0]
    pred_y = y + mv[1]


    predicted_block = prev_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
    return predicted_block



def decode_frame(residual_frame, prev_frame, mv_frame, height, width, block_size):
    decoded_frame = np.zeros_like(prev_frame)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            residual_block = residual_frame[y:y + block_size, x:x + block_size]
            predicted_b = find_predicted_block(mv_frame[(x, y)], x, y, prev_frame, block_size)

            decoded_block = residual_block + predicted_b

            decoded_frame[y:y + block_size, x:x + block_size] = decoded_block
    return decoded_frame


def decode(residual_yuv_file, mv_txt_file, block_size, decoded_yuv, height, width, frames_to_process):
    frame_size = width * height
    prev_frame = np.full((height, width), 128, dtype=np.uint8)

    with open(residual_yuv_file, 'rb') as residual_yuv_fh, open(mv_txt_file,'rt') as mv_txt_fh, open(decoded_yuv, 'wb') as decoded_fh:
        frame_index = 0
        while True:
            frame_index += 1
            y_frame = residual_yuv_fh.read(frame_size)
            mv_txt =  mv_txt_fh.readline()
            if not y_frame or frame_index > frames_to_process or not mv_txt:
                break  # End of file or end of frames
            logger.debug(f"Decoding frame {frame_index}/{frames_to_process}")
            residual_frame = np.frombuffer(y_frame, dtype=np.uint8).reshape((height, width))
            mv = parse_mv(mv_txt)

            decoded_frame = decode_frame(residual_frame, prev_frame, mv, height, width, block_size)
            write_y_only_frame(decoded_fh, decoded_frame)

            prev_frame = decoded_frame
    print('end decoding')