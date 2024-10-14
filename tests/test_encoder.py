from unittest import TestCase

import numpy as np
from fontTools.unicodedata import block

from block_predictor import predict_block
from encoder import encode, encode_frame


class Test(TestCase):
    def test_predict_block(self):
        curr_b_size = 8
        prev_b_size = curr_b_size + 2

        prev_b = np.zeros((prev_b_size, prev_b_size), dtype=np.uint8)
        curr_b = np.zeros((curr_b_size, curr_b_size), dtype=np.uint8)

        prev_b[-3:, -3:] = np.ones((3, 3))
        curr_b[-3:, -3:] = np.ones((3, 3))

        # curr_b = np.roll(curr_b, -3, axis=1)
        # curr_b = np.roll(curr_b, -3, axis=0)



        best_mv, min_mae, predicted_block = predict_block(curr_b, prev_b, 8)

        self.assertEqual(best_mv, [2,2])
        self.assertEqual(min_mae, 0)
        self.assertEqual(curr_b.all(), predicted_block.all())

        prev_b = np.roll(prev_b, -2, axis=1)
        prev_b = np.roll(prev_b, -1, axis=0)
        best_mv, min_mae, predicted_block = predict_block(curr_b, prev_b, 8)
        self.assertEqual(best_mv, [0,1])

    def test_encode_frame_1(self):
        # test middle block
        search_range = 3
        block_size = 8
        f_size = block_size * 3

        marker_fill = 69
        marker_size = 2
        marker_x_start = block_size +  2
        marker_y_start = block_size + 2

        marker_x_tx = 1
        marker_y_tx = 1

        prev_f = np.zeros((f_size, f_size), dtype=np.uint8)
        prev_f[marker_y_start:marker_y_start + marker_size, marker_x_start:marker_x_start+marker_size ] = np.full((marker_size, marker_size), marker_fill)

        curr_f = np.roll(prev_f, -1 * marker_x_tx, axis=1)
        curr_f = np.roll(curr_f, -1 * marker_y_tx, axis=0)

        # print('prev_f', '\n', prev_f)
        # print('curr_f', '\n', curr_f)

        mv_field, avg_mae, residuals, reconstructed_frame, residual_frame = encode_frame(curr_f, prev_f, block_size,
                                                                                         search_range, 0)
        # print(mv_field)

        # Test the motion vector for the block that was rolled by (2, 2)
        self.assertIn((block_size, block_size), mv_field)  # Check if block at (8, 8) has a motion vector
        print(mv_field[(block_size, block_size)])
        self.assertEqual(mv_field[(block_size, block_size)][0], -1* marker_x_tx)
        self.assertEqual(mv_field[(block_size, block_size)][1], -1* marker_y_tx)
        # self.assertEqual(mv_field[(block_size, block_size)].all(), [-1* marker_x_tx, -1 * marker_y_tx].all())  # Expect MV to point to the original block in prev_f

        # Additional validations
        self.assertAlmostEqual(avg_mae, 0)  # MAE should be 0 if the block is perfectly predicted
        self.assertTrue(np.array_equal(reconstructed_frame, curr_f))  # Reconstructed frame should match curr_f

        # self.assertEqual(best_mv, [2,2])

        # self.assertEqual(min_mae, 0)
        # self.assertEqual(curr_b.all(), predicted_block.all())

    def test_encode_frame(self):
        # Test all blocks in the frame
        search_range = 3
        block_size = 8
        num_of_blocks = 3
        f_size = block_size * num_of_blocks  # Frame size is 3x3 blocks (24x24 pixels)

        marker_fill = 69
        marker_size = 1  # The size of the marker (2x2 pixels)

        marker_x_tx = 1  # Horizontal translation
        marker_y_tx = 0  # Vertical translation


        # Iterate through all block indices (block_x_idx, block_y_idx)
        for block_x_idx in range(num_of_blocks-1):
            for block_y_idx in range(num_of_blocks-1):
                with self.subTest(block_x=block_x_idx, block_y=block_y_idx):
                    # Determine marker's initial position within the current block
                    marker_x_start = block_size * block_x_idx + 1
                    marker_y_start = block_size * block_y_idx + 2

                    # Create the previous frame with a marker in the given block
                    prev_f = np.zeros((f_size, f_size), dtype=np.uint8)
                    prev_f[marker_y_start:marker_y_start + marker_size,
                    marker_x_start:marker_x_start + marker_size] = np.full(
                        (marker_size, marker_size), marker_fill)

                    # Create the current frame by shifting the previous frame
                    curr_f = np.roll(prev_f, -1 * marker_x_tx, axis=1)  # Horizontal shift
                    curr_f = np.roll(curr_f, -1 * marker_y_tx, axis=0)  # Vertical shift

                    # Perform motion estimation with encode_frame
                    mv_field, avg_mae, residuals, reconstructed_frame, residual_frame = encode_frame(curr_f, prev_f,
                                                                                                     block_size,
                                                                                                     search_range, 0)

                    # Validate the motion vector for the current block
                    block_coords = (block_size * block_x_idx, block_size * block_y_idx)
                    self.assertIn(block_coords, mv_field, f"No motion vector for block {block_coords}")

                    # Extract the motion vector and check its correctness
                    mv = mv_field[block_coords]
                    print(
                        f"Block ({block_x_idx}, {block_y_idx}) - Expected MV: {-1 * marker_x_tx}, {-1 * marker_y_tx}, Got: {mv}")
                    self.assertEqual(mv[0], -1 * marker_x_tx, f"Incorrect horizontal MV for block {block_coords}")
                    self.assertEqual(mv[1], -1 * marker_y_tx, f"Incorrect vertical MV for block {block_coords}")

                    # Validate reconstructed frame (optional but useful)
                    self.assertTrue(np.array_equal(reconstructed_frame, curr_f),
                                    f"Reconstructed frame does not match current frame for block {block_coords}")

                    # Additional validations (optional)
                    # Check the MAE, allowing a small margin for approximations
                    self.assertLessEqual(avg_mae, 5, f"MAE too high for block {block_coords}")
