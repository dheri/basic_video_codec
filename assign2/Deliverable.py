import copy

from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics import plot_overlay_metrics

"""
Create RD plots for a fixed set of parameters (block size = 16, search range = 4, I_Period = 8).
There should be 6 curves drawn: one for the encoder of part1, 
one for each feature added in this assignment (by itself), 
and one with all four features. 
Like in Assignment 1, use the first 10 frames of Foreman for testing, 
and build your curves using at least QPs 1, 4, 7 and 10. 
It is important to include execution times as well in the comparisons.
"""


def main():
    encoder_configs = [copy.deepcopy(EncoderConfig(16, 4, 8, 1)) for _ in range(5)]
    encoder_configs[1].nRefFrames = 4
    encoder_configs[2].fracMeEnabled = True
    encoder_configs[3].fastME = True

    encoder_configs[4].nRefFrames = 4
    encoder_configs[4].fracMeEnabled = True
    encoder_configs[4].fastME = True

    params = InputParameters('../data/foreman_cif.y', 352, 288, None, 10)
    metric_files = []

    seq_names=[
        'assign1',
        '4 nRefFrames',
        'fracME',
        'fastME',
        'All enabled',
    ]

    num_of_base_files = 0

    for idx, ec in enumerate(encoder_configs):
        for qp in [1, 4, 7, 10]:
            ec.quantization_factor = qp
            params.encoder_config = ec.validate()
            # encode_video(params)
            # plot_metrics(params)
            metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())

        if idx == 0:
            num_of_base_files = len(metric_files)
            continue # skip base

        base_metric_files = metric_files[:num_of_base_files]  # Assume the first file is the base
        current_metric_files = metric_files[idx * 4: (idx + 1) * 4]
        plot_overlay_metrics(base_metric_files, current_metric_files, seq_names[idx])



if __name__ == '__main__':
    main()
