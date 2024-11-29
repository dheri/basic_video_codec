import copy

from encoder.RateControl.lookup import generate_rc_lookup
from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics


def create_lookups():
    sequences = ['foreman_cif.y']
    block_sizes =[ 16]
    for sequence in sequences:
        for block_size in block_sizes:
            create_lookup_for_seq(f"../data/{sequence}", block_size)

def create_lookup_for_seq(y_only_file, block_size):
    resolution = (352, 288)
    encoder_configs = [copy.deepcopy(EncoderConfig(
        block_size, 2, 4, quantization_factor=i, fastME=True,
        resolution=resolution,  RCflag=True, targetBR=5_140_480)) for i in range(10)]

    params = InputParameters(
        y_only_file= y_only_file,
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_configs[0],
        frames_to_process=15
    )
    metric_files = []
    num_of_base_files = 0
    skip_period = 0

    for idx, ec in enumerate(encoder_configs):
        params.encoder_config = ec.validate()
        encode_video(params)
        plot_metrics(params)
        # decode_video(params)
        metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())

        if idx == 0:
            num_of_base_files = len(metric_files)
            skip_period =  len(metric_files)
            continue # skip base

        # base_metric_files = metric_files[:num_of_base_files]  # Assume the first file is the base
        # current_metric_files = metric_files[idx * skip_period: (idx + 1) * skip_period]
        # plot_RD_curves_metrics(base_metric_files, current_metric_files, f'q = {idx}')
        # print(calculate_row_bit_budget(ec))

    # print_average_bit_count_per_block_row(metric_files, params)
    generate_rc_lookup(metric_files, params)

