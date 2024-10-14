import os


class FileIOHelper:
    def __init__(self, input_file, block_size, search_range, residual_approx_factor):

        self.input_file = input_file
        self.block_size = block_size
        self.search_range = search_range
        self.residual_approx_factor = residual_approx_factor
        self.file_identifier = f'{block_size}_{search_range}_{residual_approx_factor}'
        self.file_prefix = os.path.splitext(self.input_file)[0]

        os.makedirs(os.path.dirname(self.get_file_name(suffix='')), exist_ok=True)

    def get_file_name(self, suffix):
        return f'{self.file_prefix}/{self.file_identifier}/{suffix}'

    def get_mv_file_name(self):
        return self.get_file_name('mv.txt')

    def get_mae_csv_file_name(self):
        return self.get_file_name('mae.csv')

    def get_mae_png_file_name(self):
        return self.get_file_name('mae.png')

    def get_mc_residual_file_name(self):
        return self.get_file_name('mc_residuals.yuv')

    def get_mc_reconstructed_file_name(self):
        return self.get_file_name('mc_reconstructed.yuv')
    def get_mc_decoded_file_name(self):
        return self.get_file_name('mc_decoded.yuv')


def write_to_file(file_handle, frame_idx, data, new_line_per_block = False):
    # file_handle.write(f'\nFrame: {frame_idx}\n')
    new_line_char = f'\n' if new_line_per_block else ''
    for k in sorted(data.keys()):
        file_handle.write(f'{new_line_char}{k[0]},{k[1]}:{data[k][0]},{data[k][1]}|')
    file_handle.write('\n')


def write_y_only_frame(file_handle, reconstructed_frame):
    file_handle.write(reconstructed_frame.tobytes())
