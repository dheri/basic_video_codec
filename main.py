import assign1
import assign2
from assign1 import ex2
from assign2 import FastME
from encoder.params import EncoderConfig
from input_parameters import InputParameters

if __name__ == "__main__":
    encoder_parameters = EncoderConfig(
        block_size=8,
        search_range=8,
        quantization_factor=4,
        I_Period=99,
        nRefFrames= 4,
        fastME= True

    )

    input_params = InputParameters(
        y_only_file='data/synthetic.y',
        width=352,
        height=288,
        encoder_config=encoder_parameters,
        frames_to_process=12
    )

    assign1.ex2.save_y_frames_to_file(input_params)

    assign2.FastME.main(input_params)
