from unittest import TestCase

import numpy as np

from encoder.entropy_encoder import zigzag_order, inverse_zigzag_order, rle_encode, rle_decode, exp_golomb_encode


class Test(TestCase):
    np.set_printoptions(legacy='1.25')

    def test_zigzag_order(self):
        matrix = [
            [-31, 9, 8, 4],
            [-4, 1, 4, 0],
            [-3, 2, 4, 0],
            [4, 0, -4, 0]
        ]

        print(zigzag_order(matrix))
        print(matrix)

    def test_inv_zigzag_order(self):
        nums = np.arange(1, 17).astype(np.int8)

        inv = inverse_zigzag_order(nums, 4)
        np.set_printoptions(legacy='1.25')
        print(f"{inv}")

        actual = zigzag_order(inv)
        np.testing.assert_array_equal(nums, actual)


class Test_rel(TestCase):
    def test_rle_encode(self):
        nums = np.arange(1, 17).astype(np.int8)
        rle = rle_encode(nums)
        print(rle)

    def test_rle_encode2(self):
        nums_1 = np.arange(1, 9)
        nums_2 = np.arange(1, 5)

        # Concatenate the arrays
        numbers = np.concatenate((
            np.zeros(16, dtype=int),
            nums_1,
            np.zeros(5, dtype=int),
            nums_2,
            np.zeros(3, dtype=int),
            nums_1,
            np.zeros(95, dtype=int),

        ))

        rle = rle_encode(numbers)
        print(rle)
        np.testing.assert_array_equal(rle, [
            16,
            -8, 1, 2, 3, 4, 5, 6, 7, 8,
            5,
            -4, 1, 2, 3, 4,
            3,
            -8, 1, 2, 3, 4, 5, 6, 7, 8,
            0])
        decoded = rle_decode(rle)
        print(decoded)


class TestExpGolomb(TestCase):
    def test_exp_golomb_encode(self):
        to_enc= [ 1, 0, 0, 1, 1 ]
        for s in to_enc:
            enc = exp_golomb_encode(s)
            print(enc)
