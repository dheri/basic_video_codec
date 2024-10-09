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


def calculate_shift_value(frame_number, direction, full_loop_frames):
    if direction in ['horizontal', 'vertical']:
        shift_value = frame_number + 1  # Start from 1 and increase indefinitely
    else:  # For diagonal shifting
        shift_value = (frame_number % full_loop_frames) + 1  # Reset after a full loop

    return shift_value


def shift_frame(frame, frame_number, width, height, direction="horizontal", full_loop_frames=0):
    shift_value = calculate_shift_value(frame_number, direction, full_loop_frames)  # Get shift value with continuous acceleration

    # Apply shift based on direction
    if direction == "horizontal":
        shifted_frame = np.roll(frame, shift_value, axis=1)  # Shift along width
    elif direction == "vertical":
        shifted_frame = np.roll(frame, shift_value, axis=0)  # Shift along height
    elif direction == "diagonal":
        shifted_frame = np.roll(frame, shift_value, axis=1)  # Horizontal shift
        shifted_frame = np.roll(shifted_frame, shift_value, axis=0)  # Vertical shift

    return shifted_frame

def generate_yuv_bytestream(width, height, num_frames=None):
    """Generates the YUV byte stream for a given number of frames."""
    frames = []
    total_frames = num_frames if num_frames is not None else 100  # Default to 100 frames
    shift_value = 1
    increasing = True  # To control acceleration and deceleration
    base_frame = generate_frame(width, height, 1)

    for frame_num in range(total_frames):

        # Determine direction based on frame_num and how many frames to complete a full rotation
        if frame_num < num_frames / 3:
            direction = "horizontal"
        elif frame_num < 2 * num_frames / 3:
            direction = "vertical"
        else:
            direction = "diagonal"

        # Full loop frames can be set based on width/height
        full_loop_frames = width if direction == "horizontal" else height

        shifted_frame = shift_frame(base_frame, frame_num, width, height, direction, full_loop_frames)

        # Convert frame to bytes and store
        frames.append(shifted_frame.tobytes())

    return b''.join(frames)

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
