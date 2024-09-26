import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

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


def yuv_to_rgb(y_plane, u_plane, v_plane):
    """Convert YUV 4:4:4 to RGB using the provided CSC matrix."""
    height, width = y_plane.shape

    # Broadcast U and V planes to the same size as Y plane
    u_plane = u_plane.reshape((height, width))
    v_plane = v_plane.reshape((height, width))

    # YUV to RGB conversion matrix
    M = np.array([[1.164, 0.0, 1.596],
                  [1.164, -0.392, -0.813],
                  [1.164, 2.017, 0.0]])

    # Normalize YUV values (from 0-255 to the correct YUV range)
    y_plane = y_plane.astype(np.float32) - 16
    u_plane = u_plane.astype(np.float32) - 128
    v_plane = v_plane.astype(np.float32) - 128

    # Stack Y, U, and V into a single array (height x width x 3)
    yuv_stack = np.stack([y_plane, u_plane, v_plane], axis=-1)

    # Apply color space conversion
    rgb_stack = np.dot(yuv_stack, M.T)

    # Clip values to [0, 255] and convert to uint8
    rgb_stack = np.clip(rgb_stack, 0, 255).astype(np.uint8)

    return rgb_stack


def save_rgb_image(rgb_image, output_path):
    """Save RGB array as a PNG image."""
    img = Image.fromarray(rgb_image, 'RGB')
    img.save(output_path)

def main(input_file, output_file, width, height):
    frame_index = 0
    num_frames = calculate_num_frames(input_file, width, height)

    for y_plane, u_plane, v_plane in read_yuv420(input_file, width, height, num_frames):
        # Upscale chroma planes
        u_444, v_444 = upscale_chroma(u_plane, v_plane)
        rgb_image = yuv_to_rgb(y_plane, u_444, v_444)

        save_yuv444(f"{output_file}_{frame_index}.yuv", y_plane, u_444, v_444)
        save_rgb_image(rgb_image, f"{output_file}_{frame_index}.png")

        frame_index += 1
