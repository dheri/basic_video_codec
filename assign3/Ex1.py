import copy
import math

from assign1.ex2 import save_y_frames_to_file
from decoder import decode_video
from encoder.RateControl.lookup import generate_rc_lookup
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def create_lookups():
    sequences = [
        ('e3_CIF.y', (352, 288)),
                 ('e3_QCIF.y', (352//2, 288//2))
    ]
    block_sizes =[8, 16]
    for sequence in sequences:
        for block_size in block_sizes:
            for i_period in [1, 21]:
                create_lookup_for_seq(sequence, block_size, i_period)

def create_lookup_for_seq(sequence, block_size, i_period):
    y_only_file = f"../data/{sequence[0]}"
    resolution = sequence[1]
    encoder_configs = [copy.deepcopy(EncoderConfig(
        block_size, 2, i_period, quantization_factor=i, fastME=True,
        resolution=resolution,  RCflag=False, targetBR=10)) for i in range(int(math.log2(block_size) + 7))]

    params = InputParameters(
        y_only_file= y_only_file,
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_configs[0],
        frames_to_process=21
    )
    metric_files = []

    for idx, ec in enumerate(encoder_configs):
        params.encoder_config = ec.validate()
        encode_video(params)
        plot_metrics(params)
        decode_video(params)
        metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())

    # print_average_bit_count_per_block_row(metric_files, params)
    generate_rc_lookup(metric_files, params)

if __name__ == '__main__':
    pass
