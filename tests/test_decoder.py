from unittest import TestCase

import numpy as np

from common import find_mv_predicted_block
from encoder.PFrame import construct_frame_from_dct_and_mv, PFrame
from encoder.params import EncoderConfig
from input_parameters import InputParameters
from tests.y_generator import generate_marked_frame


class TestDecoder(TestCase):
    def test_find_predicted_block_within_bounds(self):
        # Create a 16x16 previous frame
        prev_frame = np.arange(16 * 16).reshape(16, 16)

        # Define block size
        block_size = 8

        # Motion vector that points within bounds
        x_tx = 2
        y_tx = 1
        mv = [x_tx, y_tx]  # Moving the block 2 pixels to the right and 2 pixels down
        x = 2
        y = 3

        # Get the predicted block
        predicted_block = find_mv_predicted_block(mv, x, y, prev_frame, block_size)

        # Expected block should be the section of the prev_frame starting at (2, 2)
        expected_block = prev_frame[y + y_tx:y + y_tx + block_size, x + x_tx: x + x_tx + block_size]

        # Assert that the predicted block matches the expected block
        self.assertTrue(np.array_equal(predicted_block, expected_block),
                        f"Predicted block does not match expected block for mv {mv}")

    def test_decode_frame_right_down_motion(self):
        # Parameters for the test
        search_range = 3
        block_size = 4
        num_of_blocks = 4
        f_size = block_size * num_of_blocks
        quantization_factor = 8

        marker_fill = 99
        marker_size = 2  # The size of the marker (1x1 pixels)

        marker_x_tx = 1  # Horizontal translation
        marker_y_tx = 2  # Vertical translation

        # Iterate through all block indices (block_x_idx, block_y_idx)
        for block_x_idx in range(num_of_blocks):
            for block_y_idx in range(num_of_blocks):
                with self.subTest(block_x=block_x_idx, block_y=block_y_idx):
                    encoder_parameters = EncoderConfig(block_size, search_range, 0, quantization_factor)
                    params = InputParameters(y_only_file=None, width=f_size, height=f_size,
                                             encoder_config=encoder_parameters)

                    prev_f = generate_marked_frame(f_size, block_size, block_x_idx, block_y_idx, marker_size,
                                                   marker_fill)
                    # Create the current frame by shifting the previous frame
                    curr_f = np.roll(prev_f, 1 * marker_x_tx, axis=1)  # Horizontal shift
                    curr_f = np.roll(curr_f, 1 * marker_y_tx, axis=0)  # Vertical shift

                    # Encode the frame to get motion vectors, residuals, and reconstructed frame
                    p_frame = PFrame(curr_f, prev_f)
                    encoded_frame = p_frame.encode_mc_q_dct(encoder_parameters)
                    mv_field = encoded_frame.mv_field
                    residuals_with_mc = encoded_frame.residual_frame
                    quat_dct_coffs_with_mc = encoded_frame.quantized_dct_residual_frame

                    decoded_frame = construct_frame_from_dct_and_mv(quat_dct_coffs_with_mc, prev_f, mv_field,
                                                                    encoder_parameters)

                    np.testing.assert_allclose(decoded_frame, encoded_frame.reconstructed_frame,
                                               atol=(2),
                                               err_msg=f"Decoded frame does not match reconstructed_frame_with_mc"
                                                       f" ({block_x_idx}, {block_y_idx})\n{decoded_frame}\n{encoded_frame.reconstructed_frame}")
