from unittest import TestCase

import numpy as np

from common import get_logger
from encoder.PredictionMode import PredictionMode
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame, mv_field_to_bytearray
from encoder.byte_stream_buffer import BitStreamBuffer, compare_bits

logger = get_logger()
class TestFrame(TestCase):
    def test_encoded_I_frame_data(self):
        frame = IFrame()
        coeffs_2d = np.array([[100, -50], [20, -30]], dtype=np.int16)
        frame.quantized_dct_residual_frame = coeffs_2d
        frame.prediction_data = [1,0,1,0]

        frame.generate_pre_entropy_encoded_frame_bit_stream()
        logger.info(frame.bitstream_buffer)

        self.assertEqual(1, frame.bitstream_buffer.read_bit())
        np.testing.assert_array_equal(frame.prediction_data, frame.bitstream_buffer.read_prediction_data( frame.prediction_mode, coeffs_2d.size))
        np.testing.assert_array_equal(coeffs_2d, frame.bitstream_buffer.read_quantized_coeffs( coeffs_2d.shape[0], coeffs_2d.shape[1]))


    def test_encoded_P_frame_data(self):
        frame = PFrame()
        coeffs_2d = np.array([[100, -50], [20, -30]], dtype=np.int16)
        frame.quantized_dct_residual_frame = coeffs_2d
        mv = {0: [0,0], 1:[1,1], 2:[1,1], 3: [-1,-1]}
        frame.prediction_data = mv_field_to_bytearray(mv)

        frame.generate_pre_entropy_encoded_frame_bit_stream()
        logger.info(frame.bitstream_buffer)

        self.assertEqual(0, frame.bitstream_buffer.read_bit())
        np.testing.assert_array_equal( frame.bitstream_buffer.read_prediction_data( frame.prediction_mode, coeffs_2d.size), frame.prediction_data)
        np.testing.assert_array_equal(frame.bitstream_buffer.read_quantized_coeffs( coeffs_2d.shape[0], coeffs_2d.shape[1]), coeffs_2d)



    def test_bit_stream_read_write(self):
        coeffs_2d = np.array([[0, -1], [-1,-1]])
        prediction_data = [1,1,1,1]
        assert coeffs_2d.size == len(prediction_data)
        num_blocks = len(prediction_data)

        prediction_mode = PredictionMode.INTRA_FRAME

        bitstream = BitStreamBuffer()
        bitstream.write_bit(prediction_mode.value)


        bitstream.write_prediction_data(prediction_mode, prediction_data )  # Differential info example, 4 bits
        bitstream.write_quantized_coeffs(coeffs_2d)

        bitstream.flush()
        print(bitstream)

        bitstream_buffer_copy = BitStreamBuffer()
        bitstream_buffer_copy.byte_stream = bitstream.get_bitstream().copy()

        prediction_mode_read_value = bitstream_buffer_copy.read_bit()  # Should return 1 (INTRA)
        differential_prediction_read = bitstream_buffer_copy.read_prediction_data(prediction_mode, num_blocks)
        quantized_coeffs_read = bitstream_buffer_copy.read_quantized_coeffs(coeffs_2d.shape[0], coeffs_2d.shape[1])

        self.assertEqual(prediction_mode.value, prediction_mode_read_value)
        np.testing.assert_array_equal(prediction_data, differential_prediction_read)
        np.testing.assert_array_equal(coeffs_2d, quantized_coeffs_read)

        self.assertEqual(len(bitstream.byte_stream) , 9)
        self.assertEqual(len(bitstream_buffer_copy.byte_stream) , 0)
