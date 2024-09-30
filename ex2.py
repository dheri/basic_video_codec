import os
import numpy as np
from common import calculate_num_frames

# Function to read YUV420 frames and return only Y-plane
def read_y_component(file_path, width, height, num_frames):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)  # For U and V planes, which will be skipped

    with open(file_path, 'rb') as file:
        for _ in range(num_frames):
            # Read Y plane
            y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

            # Skip U and V planes
            file.read(uv_size)
            file.read(uv_size)

            yield y_plane


# Function to save Y-only frames to individual files
def save_y_frames(input_file, output_file, width, height):
    num_frames = calculate_num_frames(input_file, width, height)
    if os.path.exists(output_file):
        print(f"y only {output_file} already exists. skipping...")
        return
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        for frame_index, y_plane in enumerate(read_y_component(input_file, width, height, num_frames)):
            f_out.write(y_plane.tobytes())




def main(input_file, output_file, width, height):
    save_y_frames(input_file, output_file, width, height)
