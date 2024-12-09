import copy

from assign1.ex2 import save_y_frames_to_file
from assign3.Ex1 import create_lookups
from decoder import decode_video
from encoder.RateControl.lookup import generate_rc_lookup
from encoder.encoder import encode_video, encode_video_parallel
from encoder.RateControl.lookup import generate_rc_lookup, rc_lookup_file_path
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def main():
    resolution = (352 // 2, 288 // 2)
    parallel = 1
    # Add ParallelMode to the EncoderConfig to support parallelism
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
        RCflag=True,
        targetBR=1_140_480,
        resolution=resolution,
        ParallelMode=1  # Set ParallelMode (1: Type 1, 2: Type 2, 3: Type 3)
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

    # Save input frames (optional preprocessing)
    save_y_frames_to_file(input_params)
    
    if parallel:
        encode_video_parallel(input_params)
    else:

    # Perform encoding based on the specified parallel mode
        encode_video(input_params)
    
    # Plot metrics to analyze performance (PSNR, bitrate, etc.)
    plot_metrics(input_params)
    
    # Decode the encoded video to validate correctness
    # save_y_frames_to_file(input_params)
    encode_video(input_params)
    plot_metrics(input_params)
    decode_video(input_params)


if __name__ == '__main__':
    # Generate required lookup tables
    create_lookups()

    # Call the main function
    # create_lookups()
    main()
