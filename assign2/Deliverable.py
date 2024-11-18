import copy

from encoder.encoder import encode_video
from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics import plot_combined_metrics

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

    for idx, ec in enumerate(encoder_configs):
        for qp in [1, 4, 7, 10]:
            ec.quantization_factor = qp
            params.encoder_config = ec.validate()
            encode_video(params)
            metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())

            # plot_metrics(params)
            # decode_video(input_params)

        plot_combined_metrics(metric_files, idx)


if __name__ == '__main__':
    main()
