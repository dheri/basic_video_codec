import os

import ex1
import ex2
import ex3

if __name__ == "__main__":
    input_file = 'data/foreman_cif.yuv'
    y_only_file = f'{os.path.splitext(input_file)[0]}.y'
    width, height = 352, 288

    # ex1.main(input_file, width, height)
    # ex2.main(input_file, width, height)

    ex2.save_y_frames(input_file,y_only_file,352, 288)
    ex3.main(input_file, width, height)

