import copy

from assign3.Ex1 import create_lookups
from decoder import decode_video
from encoder.RateControl.lookup import generate_rc_lookup
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def main():
    resolution = (352, 288)

    encoder_config = EncoderConfig(
        block_size=16,
        search_range=1,
        quantization_factor=4,
        I_Period=4,
        fastME=True,
        RCflag=True,
        targetBR=3_140_480,
        resolution=resolution
    )

    input_params = InputParameters(
        y_only_file='../data/foreman_cif.y',
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=3
    )

    encode_video(input_params)
    # plot_metrics(input_params)
    decode_video(input_params)


if __name__ == '__main__':
    # create_lookups()
    main()
