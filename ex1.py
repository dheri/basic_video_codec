import os
import numpy as np
from scipy.ndimage import zoom

def upscale_chroma(u_plane, v_plane):
    # Use bilinear interpolation to upscale U and V planes
    u_444 = zoom(u_plane, 2, order=1)
    v_444 = zoom(v_plane, 2, order=1)

    return u_444, v_444


def read_yuv420(file_path, width, height, num_frames):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    with open(file_path, 'rb') as file:
        for _ in range(num_frames):
            # Read Y plane
            y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

            # Read U and V planes
            u_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
            v_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

            yield y_plane, u_plane, v_plane


def save_yuv444(output_path, y_plane, u_plane, v_plane):
    """Saves YUV 4:4:4 data to a raw file."""
    with open(output_path, 'wb') as file:
        file.write(y_plane.tobytes())
        file.write(u_plane.tobytes())
        file.write(v_plane.tobytes())


def calculate_num_frames(file_path, width, height):
    """Calculate the number of frames in a YUV 4:2:0 file."""
    file_size = os.path.getsize(file_path)
    frame_size = width * height + 2 * (width // 2) * (height // 2)
    return file_size // frame_size


def main(input_file, output_file, width, height):
    num_frames = calculate_num_frames(input_file, width, height)
    frame_index = 0

    for y_plane, u_plane, v_plane in read_yuv420(input_file, width, height, num_frames):
        # Upscale chroma planes
        u_444, v_444 = upscale_chroma(u_plane, v_plane)

        # Save the YUV 4:4:4 frame
        save_yuv444(f"{output_file}_{frame_index}.yuv", y_plane, u_444, v_444)
        frame_index += 1
