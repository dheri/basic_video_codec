from unittest import TestCase

import numpy as np

from encoder.Frame import Frame
from encoder.PredictionMode import PredictionMode
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame
from encoder.byte_stream_buffer import BitStreamBuffer, compare_bits


class TestFrame(TestCase):
    def test_encoded_frame_data(self):
        # frame = IFrame()
        frame = PFrame()
        quant_coeffs = np.array([[100, -50], [20, -30]], dtype=np.int16)
        # quant_coeffs = np.array([0], dtype=np.int16)
        frame.quantized_dct_residual_frame = quant_coeffs

        frame.pre_entropy_encoded_frame_bit_stream()
        # frame.bitstream_buffer.buffer.append(5)
        print(frame.bitstream_buffer)
        expected = len(frame.bitstream_buffer.byte_stream) * 8
        self.assertEqual(expected, frame.encoded_frame_data_length() + 3 )
    def test_first_bit(self):
        iframe = IFrame()
        iframe.pre_entropy_encoded_frame_bit_stream()
        self.assertTrue(compare_bits(iframe.bitstream_buffer.get_bitstream(), 0, 7, 0),
                        'iframe first bit is not zero')

        pframe = PFrame()
        pframe.pre_entropy_encoded_frame_bit_stream()
        self.assertTrue(compare_bits(pframe.bitstream_buffer.get_bitstream(), 0, 7, 1),
                        'pframe first bit is not 1')


    def test_frame_bit_stream(self):

        iframe = IFrame()
        iframe.pre_entropy_encoded_frame_bit_stream()
        self.assertTrue(compare_bits(iframe.bitstream_buffer.get_bitstream(), 0, 7, 0),
                        'iframe first bit is not zero')

    def test_bit_stream_read_write(self):
        coeffs_2d = np.array([[0, -1], [0,-1]])

        bitstream = BitStreamBuffer()
        bitstream.write_bit(PredictionMode.INTRA_FRAME.value)  # Intra-frame

        differential_info = 31 # cant encode negative values in 7 bits
        differential_info_len = 7
        bitstream.write_bits(differential_info, differential_info_len)  # Differential info example, 4 bits
        bitstream.write_quantized_coeffs(coeffs_2d)

        bitstream.flush()
        print(bitstream)

        bitstream_buffer_copy = BitStreamBuffer()
        bitstream_buffer_copy.byte_stream = bitstream.get_bitstream().copy()

        prediction_mode = bitstream_buffer_copy.read_bit()  # Should return 1 (INTRA)
        differential_prediction_read = bitstream_buffer_copy.read_bits(differential_info_len)  # Should return 3
        quantized_coeffs_read = bitstream_buffer_copy.read_quantized_coeffs(coeffs_2d.size)

        self.assertEqual(PredictionMode.INTRA_FRAME.value, prediction_mode)
        np.testing.assert_array_equal(differential_info, differential_prediction_read)
        np.testing.assert_array_equal(coeffs_2d, quantized_coeffs_read)

        self.assertEqual(len(bitstream.byte_stream) , 9)
        self.assertEqual(len(bitstream_buffer_copy.byte_stream) , 0)
