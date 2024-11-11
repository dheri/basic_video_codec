import os

import numpy as np
from scipy.ndimage import zoom

from common import calculate_num_frames


# Function to upscale U and V planes
def upscale_chroma(u_plane, v_plane):
    u_444 = zoom(u_plane, 2, order=1)  # Bilinear interpolation
    v_444 = zoom(v_plane, 2, order=1)
    return u_444, v_444


# Function to read YUV420 frames
def read_yuv420(file, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # Read Y plane
    y_plane = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

    # Read U and V planes
    u_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
    v_plane = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

    return y_plane, u_plane, v_plane


# Function to convert YUV 4:4:4 to RGB
def yuv_to_rgb(y_plane, u_plane, v_plane):
    height, width = y_plane.shape

    # Broadcast U and V planes to the same size as Y plane
    u_plane = u_plane.reshape((height, width))
    v_plane = v_plane.reshape((height, width))

    # YUV to RGB conversion matrix
    M = np.array([[1.164, 0.0, 1.596],
                  [1.164, -0.392, -0.813],
                  [1.164, 2.017, 0.0]])

    # Normalize YUV values
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


def create_noise_mask(height, width, noise_percent):
    mask = np.zeros((height, width), dtype=np.uint8)
    num_noise_pixels = int((noise_percent / 100.0) * height * width)
    noise_indices = np.random.choice(height * width, num_noise_pixels, replace=False)
    mask.flat[noise_indices] = 1
    return mask


def generate_random_values(height, width):
    return np.random.randint(0, 256, size=(height, width), dtype=np.uint8)


def apply_mask(channel, mask, strategy='randomize', random_values=None):
    if strategy == 'turn_off':
        return np.where(mask == 1, 0, channel)
    elif strategy == 'flip_value':
        return np.where(mask == 1, 128 + (128 - channel), channel)
    elif strategy == 'randomize':
        if random_values is None:
            raise ValueError('random_values must be provided when strategy is randomize')
        return np.where(mask == 1, random_values, channel)  # Use the precomputed random values
    return channel


def pad_or_resize_channel(channel, target_height, target_width):
    """Resize or pad the channel to fit the target size."""
    current_height, current_width = channel.shape
    if current_height != target_height or current_width != target_width:
        # Resize or pad the channel to fit the grid unit size
        padded_channel = np.zeros((target_height, target_width), dtype=np.uint8)

        # Fit the channel within the padded space (centered)
        start_y = (target_height - current_height) // 2
        start_x = (target_width - current_width) // 2

        padded_channel[start_y:start_y + current_height, start_x:start_x + current_width] = channel
        return padded_channel
    return channel


def construct_grid(channels, masks, original_width, original_height, half_border, spacing, random_values):
    grid_unit_x = int(original_width * 1.0625)
    grid_unit_y = int(original_height * 1.0625)

    num_masks = len(masks)
    grid_height = (num_masks * grid_unit_y) + ((num_masks - 1) * spacing) + 2 * half_border
    grid_width = (len(channels) * grid_unit_x) + ((len(channels) - 1) * spacing) + 2 * half_border

    # print(f"Total grid dimensions: Height = {grid_height}, Width = {grid_width}")
    # Initialize the grid buffer
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    for i, (channel_name, channel_data) in enumerate(channels.items()):
        for j, mask in enumerate(masks):
            # Apply the mask to the channel
            masked_channel = apply_mask(channel_data, mask, 'randomize', random_values)

            # Resize or pad the masked channel to fit the grid unit size
            padded_channel = pad_or_resize_channel(masked_channel, grid_unit_y, grid_unit_x)

            # Calculate position in grid (shift down for each channel)
            start_x = half_border + i * (grid_unit_x + spacing)  # Shift right for each channel
            start_y = half_border + j * (grid_unit_y + spacing)  # Shift down for each mask

            # Place the padded channel into the grid
            grid[start_y:start_y + grid_unit_y, start_x:start_x + grid_unit_x] = padded_channel

    return grid


def main(input_file, width, height):
    file_prefix = os.path.splitext(input_file)[0]
    output_file = f'{file_prefix}_noizy.yuv'
    num_frames = calculate_num_frames(input_file, width, height)
    noise_percents = [0, 1, 2, 4, 90]
    masks = [create_noise_mask(height, width, p) for p in noise_percents]
    random_values = generate_random_values(height, width)

    grid_unit_size = width
    half_border = grid_unit_size // 8

    grid_unit_x = int(width * 1.0625)
    grid_unit_y = int(height * 1.0625)
    num_masks = len(masks)

    channels = ['M', 'Y', 'U', 'V', 'R', 'G', 'B']
    spacing = 0

    grid_height = (num_masks * grid_unit_y) + ((num_masks - 1) * spacing) + 2 * half_border
    grid_width = (len(channels) * grid_unit_x) + ((len(channels) - 1) * spacing) + 2 * half_border
    print(f"{output_file} dimensions: Height = {grid_height}, Width = {grid_width}")

    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        for frame_index in range(num_frames):
            try:
                # Read one frame (YUV420)
                y_plane, u_plane, v_plane = read_yuv420(f_in, width, height)

                # Upscale chroma planes
                u_444, v_444 = upscale_chroma(u_plane, v_plane)
                rgb_image = yuv_to_rgb(y_plane, u_444, v_444)

                # Prepare channels
                channels = {
                    'M': np.ones((height, width), dtype=np.uint8) * 128,
                    'Y': y_plane,
                    'U': u_444,
                    'V': v_444,
                    'R': rgb_image[:, :, 0],  # Red channel from RGB
                    'G': rgb_image[:, :, 1],  # Green channel from RGB
                    'B': rgb_image[:, :, 2]   # Blue channel from RGB
                }

                # Construct the grid with masked channels
                grid = construct_grid(channels, masks, width, height, half_border, spacing, random_values)

                # Write grid to the output file
                f_out.write(grid.tobytes())

            except Exception as e:
                print(f"Error processing frame {frame_index}: {e}")
                break
