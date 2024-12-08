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
    resolution = (352 // 2, 288 // 2)

    # Add ParallelMode to the EncoderConfig to support parallelism
    encoder_config = EncoderConfig(
        block_size=16,
        search_range=1,
        quantization_factor=4,
        I_Period=21,
        fastME=True,
        RCflag=True,
        targetBR=1_140_480,
        resolution=resolution,
        ParallelMode=1  # Set ParallelMode (1: Type 1, 2: Type 2, 3: Type 3)
    )

    input_params = InputParameters(
        y_only_file='../data/e3_QCIF.y',
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=6
    )

    # Save input frames (optional preprocessing)
    save_y_frames_to_file(input_params)
    
    # Perform encoding based on the specified parallel mode
    encode_video(input_params)
    
    # Plot metrics to analyze performance (PSNR, bitrate, etc.)
    plot_metrics(input_params)
    
    # Decode the encoded video to validate correctness
    decode_video(input_params)


if __name__ == '__main__':
    # Generate required lookup tables
    create_lookups()

    # Call the main function
    main()
