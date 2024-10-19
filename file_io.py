import os

from input_parameters import InputParameters


class FileIOHelper:
    def __init__(self, params:InputParameters):

        self.y_only_file = params.y_only_file
        self.block_size = params.encoder_parameters.block_size
        self.search_range = params.encoder_parameters.search_range
        self.residual_approx_factor = params.encoder_parameters.residual_approx_factor
        self.frames_to_process = params.frames_to_process

        self.file_identifier = f'{self.block_size}_{self.search_range}_{self.residual_approx_factor}'
        self.file_prefix = os.path.splitext(self.y_only_file)[0]

        os.makedirs(os.path.dirname(self.get_file_name(suffix='')), exist_ok=True)

    def get_file_name(self, suffix):
        return f'{self.file_prefix}/{self.file_identifier}/{suffix}'

    def get_file_name_wo_identifier(self, suffix):
        return f'{self.file_prefix}/{suffix}'

    def get_y_file_name(self):
        return  f'{self.file_prefix}.y'

    def get_yuv_file_name(self):
        return  f'{self.file_prefix}.yuv'

    def get_mv_file_name(self):
        return self.get_file_name('mv.txt')

    def get_metrics_csv_file_name(self):
        return self.get_file_name('metrics.csv')

    def get_metrics_png_file_name(self):
        return self.get_file_name('metrics.png')

    def get_mc_residual_file_name(self):
        return self.get_file_name('mc_residuals.yuv')

    def get_mc_reconstructed_file_name(self):
        return self.get_file_name('mc_reconstructed.yuv')
    def get_mc_decoded_file_name(self):
        return self.get_file_name('mc_decoded.yuv')


def write_mv_to_file(file_handle, data, new_line_per_block = False):
    # file_handle.write(f'\nFrame: {frame_idx}\n')
    new_line_char = f'\n' if new_line_per_block else ''
    for k in sorted(data.keys()):
        file_handle.write(f'{new_line_char}{k[0]},{k[1]}:{data[k][0]},{data[k][1]}|')
    file_handle.write('\n')


def write_y_only_frame(file_handle, reconstructed_frame):
    file_handle.write(reconstructed_frame.tobytes())
