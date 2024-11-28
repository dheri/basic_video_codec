import copy

from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_overlay_metrics, plot_metrics


def main():
    encoder_configs = [copy.deepcopy(EncoderConfig(16, 4, 8, quantization_factor=i)) for i in range(10)]

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


if __name__ == '__main__':
    main()
