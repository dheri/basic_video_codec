import copy

from encoder.params import EncoderConfig
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics.metrics import plot_rd_curves, calculate_rd_points_and_times, tabulate_and_export_encoding_times, \
    plot_per_frame_psnr


def run_experiments():
    y_only_file = f"../data/e3_CIF.y"
    resolution = (352, 288)
    encoder_config = EncoderConfig( 16, 2, 21, quantization_factor=6, fastME=True, fracMeEnabled=True,
        resolution=resolution, )

    params = InputParameters(
        y_only_file= y_only_file,
        width=resolution[0],
        height=resolution[1],
        encoder_config=encoder_config,
        frames_to_process=21
    )

    bit_rates=[7_000_000, 2_400_000, 360_000]

    series_collection = []
    encoding_times_collection = []
    metric_files = []
    two_m_metric_files = []
    for qp in [3,6,9]:
        ec = copy.deepcopy(encoder_config)
        ec.quantization_factor = qp
        params.encoder_config = ec.validate()
        run_encoder(params)
        metric_files.append(FileIOHelper(params).get_metrics_csv_file_name())
    rd_points, encoding_times = calculate_rd_points_and_times(metric_files, f"RCMode: 0")
    series_collection.append(rd_points)
    encoding_times_collection.append(encoding_times)
    metric_files.clear()

    for rc_mode in range(1,4)[:]:
        for bit_rate in bit_rates[:]:
            ec = copy.deepcopy(encoder_config)
            ec.RCflag = rc_mode
            ec.targetBR = bit_rate
            params.encoder_config = ec.validate()
            run_encoder(params)
            metric_file_name  = FileIOHelper(params).get_metrics_csv_file_name()
            metric_files.append(metric_file_name)
            if bit_rate == 2_400_000:
                two_m_metric_files.append(metric_file_name)
            rd_points, encoding_times = calculate_rd_points_and_times(metric_files, f"RCMode: {rc_mode}")
        series_collection.append(rd_points)
        encoding_times_collection.append(encoding_times)

        metric_files.clear()

    plot_rd_curves(series_collection)
    tabulate_and_export_encoding_times(series_collection, encoding_times_collection)


    plot_per_frame_psnr(two_m_metric_files)


def run_encoder(params: InputParameters):

        # encode_video(params)
        # plot_metrics(params)
        # decode_video(params)
        pass


if __name__ == '__main__':
    run_experiments()
    pass
