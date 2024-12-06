import copy

from assign1.ex2 import save_y_frames_to_file
from assign3.Ex1 import create_lookups
from decoder import decode_video
from encoder.RateControl.lookup import generate_rc_lookup
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def main():
    resolution = (352//2, 288//2)

    encoder_config = EncoderConfig(
        block_size=16,
        search_range=1,
        quantization_factor=4,
        I_Period=21,
        fastME=True,
        RCflag=True,
        targetBR=1_140_480,
        resolution=resolution
    )

    input_params = InputParameters(
        y_only_file='../data/e3_QCIF.y',
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=6
    )

    save_y_frames_to_file(input_params)
    # encode_video(input_params)
    # plot_metrics(input_params)
    # decode_video(input_params)


if __name__ == '__main__':
    create_lookups()
    # main()
