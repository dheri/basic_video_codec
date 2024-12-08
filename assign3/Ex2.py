import copy
import math


from decoder import decode_video
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def run_experiments():
    y_only_file = f"../data/e3_CIF.y"
    resolution = (352, 288)
    encoder_config = EncoderConfig( 16, 2, 21, quantization_factor=6, fastME=True, fracMeEnabled=True,
        resolution=resolution, )

    params = InputParameters(
        y_only_file= y_only_file,
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=21
    )

    bit_rates=[7_000_000, 2_400_000, 360_000]


    for qp in [3,6,9]:
        ec = copy.deepcopy(encoder_config)
        ec.quantization_factor = qp
        params.encoder_config = ec.validate()
        run_encoder(params)

    for rc_mode in range(1,4):
        for bit_rate in bit_rates:
            ec = copy.deepcopy(encoder_config)
            ec.RCflag = rc_mode
            ec.targetBR = bit_rate
            params.encoder_config = ec.validate()
            run_encoder(params)


def run_encoder(params: InputParameters):

        # encode_video(params)
        plot_metrics(params)
        # decode_video(params)


if __name__ == '__main__':
    run_experiments()
    pass
