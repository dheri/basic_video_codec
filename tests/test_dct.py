from unittest import TestCase

import numpy as np

from encoder.dct import transform_quantize_rescale_inverse, generate_quantization_matrix


class TestDCT(TestCase):
    def test_transform_quantize_rescale_inverse(self):
        i = 8
        block =  np.random.randint(0, 255, size=(i,i))
        # block = np.zeros((8,8))
        block[1:, 1] = 5

        qp = 1
        reconstructed_block = transform_quantize_rescale_inverse(block, qp)
        np.testing.assert_allclose(block, reconstructed_block ,atol=2**(qp+2))

        qp = 5
        reconstructed_block = transform_quantize_rescale_inverse(block, qp)
        np.testing.assert_allclose(block, reconstructed_block ,atol=2**(qp+2))

    def test_generate_quantization_matrix(self):
        Q = generate_quantization_matrix(4,2)
        expected_q = np.array([[4, 4, 4, 8], [4, 4, 8, 16], [4, 8, 16, 16], [8, 16, 16, 16]])
        np.testing.assert_array_equal(Q, expected_q)

        Q = generate_quantization_matrix(2,0)
        expected_q = np.array([[1, 2],[2, 4]])
        np.testing.assert_array_equal(Q, expected_q)
