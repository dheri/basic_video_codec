import logging


import ex4
from input_parameters import InputParameters
from encoder.params import EncoderParameters
from tests.y_generator import generate_sample_file


if __name__ == "__main__":

    encoder_parameters = EncoderParameters(
        block_size = 8,
        search_range=2,
        i_period=1,
        quantization_factor=1,
    )

    input_params = InputParameters(
        y_only_file ='data/foreman_cif.y',
        width  = 352,
        height= 288,
        encoder_parameters= encoder_parameters,
        frames_to_process = 12
    )

    # ex2.main(input_params)
    # generate_sample_file(input_file, num_frames=300)

    # ex2.save_y_frames_to_file(input_params)

    # ex3.main(input_params)

    ex4.main(input_params)

