import os

import ex1
import ex2
import ex3
from tests.y_generator import generate_sample_file

if __name__ == "__main__":
    input_file = 'data/sample_video.yuv'
    y_only_file = f'{os.path.splitext(input_file)[0]}.y'
    width, height = 352, 288

    # ex1.main(input_file, width, height)
    # ex2.main(input_file, width, height)

    # generate_sample_file('data/sample_video.yuv', num_frames=200)
    ex2.save_y_frames(input_file,y_only_file,352, 288)
    ex3.main(input_file, width, height)

