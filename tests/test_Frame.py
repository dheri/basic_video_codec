from unittest import TestCase

import numpy as np

from common import get_logger
from encoder.PredictionMode import PredictionMode
from encoder.IFrame import IFrame
from encoder.PFrame import PFrame, mv_field_to_bytearray, byte_array_to_mv_field
from encoder.byte_stream_buffer import BitStreamBuffer, compare_bits
from encoder.params import EncoderConfig
from input_parameters import InputParameters

logger = get_logger()
class TestFrame(TestCase):
    def test_encoded_I_frame_data(self):
        frame = IFrame()
        coeffs_2d = np.array([[100, -50], [20, -30]], dtype=np.int16)
        frame.quantized_dct_residual_frame = coeffs_2d
        frame.prediction_data = [1,0,1,0]
        encoder_config = EncoderConfig(block_size=2, search_range=1,I_Period=1, quantization_factor=0)

        frame.generate_pre_entropy_encoded_frame_bit_stream(encoder_config)
        logger.info(frame.bitstream_buffer)

        self.assertEqual(1, frame.bitstream_buffer.read_bit())
        np.testing.assert_array_equal(frame.prediction_data, frame.bitstream_buffer.read_prediction_data( frame.prediction_mode, coeffs_2d.size))
        np.testing.assert_array_equal(coeffs_2d, frame.bitstream_buffer.read_quantized_coeffs( coeffs_2d.shape[0], coeffs_2d.shape[1]))


    def test_encoded_P_frame_data(self):
        frame = PFrame()
        encoder_config = EncoderConfig(block_size=2, search_range=1,I_Period=1, quantization_factor=0)
        params = InputParameters(height=2, width=2, encoder_config=encoder_config, y_only_file=None)


        coeffs_2d = np.array([[100, -50], [20, -30]], dtype=np.int16)
        frame.quantized_dct_residual_frame = coeffs_2d
        mv = {0: [0,0], 1:[1,1], 2:[1,1], 3: [-1,-1]}
        frame.prediction_data = mv_field_to_bytearray(mv)

        frame.generate_pre_entropy_encoded_frame_bit_stream(encoder_config)
        
        decoded_frame = PFrame()
        decoded_frame.construct_frame_metadata_from_bit_stream(params,frame.bitstream_buffer.byte_stream.copy())
        frame.parse_prediction_data(params)
        logger.info(frame.bitstream_buffer)

        self.assertEqual(0, frame.bitstream_buffer.read_bit())
        np.testing.assert_array_equal( frame.bitstream_buffer.read_prediction_data( frame.prediction_mode, coeffs_2d.size), frame.mv_field)
        np.testing.assert_array_equal(frame.bitstream_buffer.read_quantized_coeffs( coeffs_2d.shape[0], coeffs_2d.shape[1]), coeffs_2d)

    def test_mv_field_to_bytearray_and_back(self):
        # Sample motion vector field
        mv_field = {
            (0, 0): [0, 1],
            (4, 0): [1, -1],
            (8, 0): [-1, -1],
            (0, 4): [-1, 0],
            (4, 4): [2, 2],
            (8, 4): [2, -1]
        }

        # Expected byte stream after conversion
        expected_byte_stream = bytearray([0, 1, 1, 255, 255, 255, 255, 0, 2, 2, 2, 255])  # Corresponding unsigned bytes

        # Convert motion vector field to byte array
        byte_stream = mv_field_to_bytearray(mv_field)

        # Assert the byte stream matches the expected output
        self.assertEqual(byte_stream, expected_byte_stream)

        # Convert byte array back to motion vector field
        reconstructed_mv_field = byte_array_to_mv_field(byte_stream, width=12, height=8, block_size=4)

        # Assert the reconstructed motion vector field matches the original
        self.assertEqual(reconstructed_mv_field, mv_field)

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
