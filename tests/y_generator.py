import numpy as np

def generate_circle_quadrant(width, height):
    """Generates a smooth gradient circle quadrant with a radial fade, placed in the top-right corner."""
    radius = int(width / 3)  # Radius is 1/3 of the width
    x_center = width  # x center is at the very right edge of the frame
    y_center = 0  # y center is at the top edge of the frame (for top-right corner placement)

    y, x = np.ogrid[:height, :width]  # Create grid for height and width

    # Calculate distance from the circle's center
    distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Create a mask for points within the radius of the circle
    circle_mask = distance_from_center <= radius

    # Initialize the circle area with zeros (background)
    circle = np.zeros((height, width), dtype=np.uint8)

    # Apply the gradient: smooth fade from white to black and back to white
    circle[circle_mask] = np.clip(255 - (255 * distance_from_center[circle_mask] / radius), 0, 255).astype(np.uint8)

    return circle


def generate_triangle(width, height):
    """Generates a triangle with an approximate gradient."""
    base = width // 3
    triangle = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if x >= width - base and y >= height - (x - (width - base)):
                distance_to_edge = min(abs(x - (width - base)), abs(y - (height - base)))
                triangle[y, x] = np.clip(255 - (255 * distance_to_edge / base), 0, 255).astype(np.uint8)
    return triangle

def generate_checkerboard(width, height):
    """Generates a checkerboard pattern with 32x32 squares, subdivided accordingly."""
    min_block_size = 32
    img = np.zeros((height, width), dtype=np.uint8)

    block_size = min_block_size
    while block_size <= min_block_size * 8:
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    img[y:y + block_size, x:x + block_size] = 255
        block_size *= 2
    return img

def generate_frame(width, height, frame_number):
    """Generates a single Y-frame for a given frame number."""
    frame = np.zeros((height, width), dtype=np.uint8)

    # Add circle quadrant (top right)
    circle = generate_circle_quadrant(width, height)
    frame = np.maximum(frame, circle)

    # Add triangle (lower right)
    triangle = generate_triangle(width, height)
    frame = np.maximum(frame, triangle)

    # Add checkerboard (rest of the screen)
    # checkerboard = generate_checkerboard(width, height)
    # frame = np.maximum(frame, checkerboard)

    # Add frame counter in binary (left middle third)
    # bin_counter = f"{frame_number:032b}"  # Display the counter in binary
    # counter_height = height // 3
    # for i, bit in enumerate(bin_counter[:counter_height]):
    #     if bit == "1":
    #         frame[height // 3 + i, :width // 4] = 255

    return frame


def calculate_shift_value(frame_number):
    """
    Calculate the shift value based on a predefined sequence:
    0, 2, 4, 8, 16, 32, 32, 15, 7, 3, 1, and then repeat.

    Args:
    - frame_number: Current frame number.

    Returns:
    - shift_value: The calculated pixel shift for this frame.
    """
    # Define the shift pattern
    shift_pattern = [0, 2, 4, 8, 16, 32, 32, 15, 7, 3, 1]

    # Calculate the shift based on the frame number, cycling through the pattern
    shift_value = shift_pattern[frame_number % len(shift_pattern)]

    return shift_value


def shift_frame(frame, shift_value,  direction="horizontal", ):
    # Apply shift based on direction
    shifted_frame = None
    if direction == "horizontal":
        shifted_frame = np.roll(frame, shift_value, axis=1)  # Shift along width
    elif direction == "vertical":
        shifted_frame = np.roll(frame, shift_value, axis=0)  # Shift along height
    elif direction == "diagonal":
        shifted_frame = np.roll(frame, shift_value, axis=1)  # Horizontal shift
        shifted_frame = np.roll(shifted_frame, shift_value, axis=0)  # Vertical shift

    return shifted_frame

def generate_video_stream(base_frame, width, height, num_frames=None):
    # Total frames required for full horizontal, vertical, and diagonal loops
    total_frames_for_full_loop = width + height + min(width, height)
    total_frames = num_frames if num_frames is not None else total_frames_for_full_loop

    byte_stream = bytearray()
    frame_num = 0

    # Define the order of directions
    directions = ["horizontal", "vertical", "diagonal"]
    direction_idx = 0
    direction = directions[direction_idx]

    current_direction_length = 0  # Track how far we are in the current direction
    shift_value = 0
    prev_frame = base_frame
    while frame_num < total_frames:
        # Update direction based on the current progress in the loop
        if direction == "horizontal" and current_direction_length >= width:
            direction_idx += 1
            direction = directions[direction_idx]
            current_direction_length = 0  # Reset for the next direction
        elif direction == "vertical" and current_direction_length >= height:
            direction_idx += 1
            direction = directions[direction_idx]
            current_direction_length = 0  # Reset for the next direction
        elif direction == "diagonal" and current_direction_length >= min(width, height):
            direction_idx = 0  # Loop back to horizontal
            direction = directions[direction_idx]
            current_direction_length = 0

        # Calculate the shift value based on the frame number
        shift_value = calculate_shift_value(frame_num)
        # Shift the base frame by the calculated value
        shifted_frame = shift_frame(prev_frame, shift_value, direction)
        prev_frame = shifted_frame
        # Convert the shifted frame to bytes and append to the byte stream
        byte_stream.extend(shifted_frame.tobytes())

        # Increment frame count and current direction length
        frame_num += 1
        current_direction_length += shift_value  # Progress based on shift value

    return bytes(byte_stream)


def generate_yuv_bytestream(width, height, num_frames=None):
    base_frame = generate_frame(width, height, 1)
    return generate_video_stream(base_frame, width, height, num_frames)


def save_yuv_to_file(filename, width, height, num_frames=None):
    """Saves the generated YUV byte stream to a file."""
    byte_stream = generate_yuv_bytestream(width, height, num_frames)
    with open(filename, 'wb') as f:
        f.write(byte_stream)

# Example usage:

def generate_sample_file(filename, width = 352, height=288, num_frames = 12):
    save_yuv_to_file(filename, width, height, num_frames)

def generate_sample_vide_bitstream(width = 352, height=288, num_frames = 12):
    generate_yuv_bytestream(width, height, num_frames)
