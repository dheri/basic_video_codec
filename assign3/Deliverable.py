import copy

from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_metrics
from metrics.plot_rd_curves import plot_RD_curves_metrics
from metrics.bit_count_per_block import print_average_bit_count_per_block_row


def main():
    encoder_configs = [copy.deepcopy(EncoderConfig(4, 4, 3, quantization_factor=i)) for i in range(10)]

    params = InputParameters(
        y_only_file='../data/synthetic.y',
        width=352,
        height=288,
        encoder_config=encoder_configs[0],
        frames_to_process=5
    )
    metric_files = []
    num_of_base_files = 0
    skip_period = 0

    for idx, ec in enumerate(encoder_configs):
        params.encoder_config = ec.validate()
        # encode_video(params)
        # plot_metrics(params)
        # decode_video(params)
        metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())

        if idx == 0:
            num_of_base_files = len(metric_files)
            skip_period =  len(metric_files)
            continue # skip base

        # base_metric_files = metric_files[:num_of_base_files]  # Assume the first file is the base
        # current_metric_files = metric_files[idx * skip_period: (idx + 1) * skip_period]
        # plot_RD_curves_metrics(base_metric_files, current_metric_files, f'q = {idx}')
    print_average_bit_count_per_block_row(metric_files, params)

if __name__ == '__main__':
    main()
