import copy

from assign1.ex2 import save_y_frames_to_file
from assign3.Ex1 import create_lookups
from decoder import decode_video
from encoder.RateControl.lookup import generate_rc_lookup, rc_lookup_file_path
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def main():
    seq_idx = 0
    sequences = [
        ('e3_CIF.y', (352, 288), 24_00_000),
        # ('e3_QCIF.y', (176,144),   960_000)
    ]
    targetBR = 24_00_000
    resolution = sequences[seq_idx][1]
    encoder_config = EncoderConfig(
        block_size=16,
        search_range=1,
        quantization_factor=5,
        I_Period=21,
        fastME=True,
        RCflag=3,
        targetBR=targetBR,
        resolution=resolution
    )


    input_params = InputParameters(
        y_only_file=f'../data/{sequences[seq_idx][0]}',
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=21
    )

    # save_y_frames_to_file(input_params)
    encode_video(input_params)
    plot_metrics(input_params)
    decode_video(input_params)


if __name__ == '__main__':
    # create_lookups()
    main()
