import os


def write_to_file(file_handle, frame_idx, data, new_line_per_block = False):
    # file_handle.write(f'\nFrame: {frame_idx}\n')
    new_line_char = f'\n' if new_line_per_block else ''
    for k in sorted(data.keys()):
        file_handle.write(f'{new_line_char}{k[0]},{k[1]}:{data[k][0]},{data[k][1]}|')
    file_handle.write('\n')


def write_y_only_frame(file_handle, reconstructed_frame):
    file_handle.write(reconstructed_frame.tobytes())

def get_file_name(original_file, suffix,  block_size, search_range, residual_approx_factor, ):
    file_prefix = os.path.splitext(original_file)[0]
    file_identifier = f'{block_size}_{search_range}_{residual_approx_factor}'
    return f'{file_prefix}/{file_identifier}/{suffix}'