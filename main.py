import os

import ex1
import ex2
import ex3
from input_parameters import InputParameters
from tests.y_generator import generate_sample_file

if __name__ == "__main__":

    # ex1.main(input_file, width, height)
    # ex2.main(input_file, width, height)

    # generate_sample_file(input_file, num_frames=300)

    input_params = InputParameters(
        y_only_file ='data/foreman_cif.y',
        width  = 352,
        height= 288,
        block_size = 8,
        search_range = 2,
        residual_approx_factor= 3,
        frames_to_process = 25
    )
    ex2.save_y_frames_to_file(input_params)
    ex3.main(input_params)

