from unittest import TestCase

import numpy as np
from fontTools.unicodedata import block

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
        frame.prediction_data = [1, 0, 1, 0]
        encoder_config = EncoderConfig(block_size=2, search_range=1, I_Period=1, quantization_factor=0)

        frame.populate_bit_stream_buffer(encoder_config)
        logger.info(frame.bitstream_buffer)

        self.assertEqual(1, frame.bitstream_buffer.read_bit())
        np.testing.assert_array_equal(frame.prediction_data,
                                      frame.bitstream_buffer.read_prediction_data(frame.prediction_mode,
                                                                                  coeffs_2d.size))
        np.testing.assert_array_equal(coeffs_2d, frame.bitstream_buffer.read_quantized_coeffs(coeffs_2d.shape[0],
                                                                                              coeffs_2d.shape[1]))

    def test_encoded_P_frame_data(self):
        encoder_config = EncoderConfig(block_size=2, search_range=1, I_Period=1, quantization_factor=0)
        params = InputParameters(height=2, width=2, encoder_config=encoder_config, y_only_file=None)

        frame_to_encode = PFrame()

        coeffs_2d = np.array([[100, -50], [20, -30]], dtype=np.int16)
        frame_to_encode.quantized_dct_residual_frame = coeffs_2d
        # mv = {(0, 0): [0, 0], (0, 2): [1, 1], (2, 0): [1, 1], (2, 2): [-1, -1]}
        mv = {(0, 0): [-1, 1]}
        frame_to_encode.mv_field = mv

        frame_to_encode.populate_bit_stream_buffer(encoder_config)

        # check prediction_data is generated of correct length
        self.assertEqual(len(frame_to_encode.prediction_data), len(mv) * 2)
        logger.info(f"frame to encode : {frame_to_encode.bitstream_buffer}")


        decoded_frame = PFrame()
        byte_stream_copy = frame_to_encode.bitstream_buffer.bit_stream.copy()

        decoded_frame.construct_frame_metadata_from_bit_stream(params, byte_stream_copy)


        np.testing.assert_array_equal(decoded_frame.quantized_dct_residual_frame, frame_to_encode.quantized_dct_residual_frame)
        logger.info(f"quantized_dct_residual_frame matched ")

        np.testing.assert_array_equal(decoded_frame.mv_field, frame_to_encode.mv_field)
        logger.info(f"mv_field  matched ")



        self.assertEqual(mv, decoded_frame.mv_field)
        logger.info(f"decoded  mv  matched {decoded_frame.mv_field}")

        # self.assertEqual(0, frame_to_encode.bitstream_buffer.read_bit())
        np.testing.assert_array_equal(coeffs_2d, decoded_frame.quantized_dct_residual_frame)
    def test_encoded_frame(self):

        # curr_frame = np.array([[100, 50], [20, 30]], dtype=np.int8)
        # prev_frame = np.array([[24, 121], [75, 23]], dtype=np.int8)

        curr_frame = np.random.randint(25, 230, size=(32, 48))
        noise = np.random.randint(-20, 20, size=curr_frame.shape, dtype=np.int16)
        prev_frame = curr_frame.astype(np.int16) + noise

        encoder_config = EncoderConfig(block_size=8, search_range=1, I_Period=1, quantization_factor=0)
        params = InputParameters(height=curr_frame.shape[0], width=curr_frame.shape[1], encoder_config=encoder_config, y_only_file=None)

        frame_to_encode = PFrame()



        frame_to_encode.curr_frame = curr_frame
        frame_to_encode.prev_frame = prev_frame

        frame_to_encode.encode(encoder_config)
        frame_to_encode.populate_bit_stream_buffer(params.encoder_config)

        # check prediction_data is generated of correct length
        # self.assertEqual(len(frame_to_encode.prediction_data), len(mv) * 2)
        logger.info(f"frame to encode : {frame_to_encode.bitstream_buffer}")


        decoded_frame = PFrame()
        decoded_frame.prev_frame = frame_to_encode.prev_frame
        byte_stream_copy = frame_to_encode.bitstream_buffer.bit_stream.copy()
        decoded_frame.construct_frame_metadata_from_bit_stream(params, byte_stream_copy)

        df = decoded_frame.decode(None, encoder_config)

        np.testing.assert_array_equal(decoded_frame.quantized_dct_residual_frame, frame_to_encode.quantized_dct_residual_frame)
        logger.info(f"quantized_dct_residual_frame  matched ")

        np.testing.assert_array_equal(decoded_frame.mv_field, frame_to_encode.mv_field)
        logger.info(f"mv_field  matched ")


        # self.assertEqual(len(decoded_frame.prediction_data), len(mv) * 2)
        # logger.info(f"decoded  decoded_frame.reconstructed_frame    {df}")

        np.testing.assert_allclose(frame_to_encode.curr_frame, df, atol=2)

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
        bs=8
        coeffs_2d = np.random.randint(-300, 230, size=(32, 48)).astype(np.int16)
        actual_prediction_data = np.random.randint(0, 1, size=coeffs_2d.size // (bs**2) ).astype(np.uint8)
        assert coeffs_2d.size // (bs**2) == len(actual_prediction_data)

        encoder_config = EncoderConfig(block_size=bs, search_range=1, I_Period=1, quantization_factor=3)
        params = InputParameters(height=coeffs_2d.shape[0], width=coeffs_2d.shape[1], encoder_config=encoder_config, y_only_file=None)


        prediction_mode = PredictionMode.INTRA_FRAME

        bitstream = BitStreamBuffer()
        bitstream.write_bit(prediction_mode.value)

        bitstream.write_prediction_data(prediction_mode, actual_prediction_data)  # Differential info example, 4 bits
        bitstream.write_quantized_coeffs(coeffs_2d, 1)

        bitstream.flush()
        print(bitstream)

        bitstream_buffer_copy = BitStreamBuffer()
        bitstream_buffer_copy.bit_stream = bitstream.bit_stream.copy()

        prediction_mode_read_value = bitstream_buffer_copy.read_bit()  # Should return 1 (INTRA)
        self.assertEqual(prediction_mode.value, prediction_mode_read_value)

        differential_prediction_read = bitstream_buffer_copy.read_prediction_data(prediction_mode, params)
        np.testing.assert_array_equal(differential_prediction_read, actual_prediction_data)

        quantized_coeffs_read = bitstream_buffer_copy.read_quantized_coeffs(coeffs_2d.shape[0], coeffs_2d.shape[1], params.encoder_config.block_size).astype(np.int16)
        np.testing.assert_array_equal(coeffs_2d, quantized_coeffs_read)


        self.assertEqual(len(bitstream.bit_stream)/8, 9)


    def test_construct_frame_metadata_from_bit_stream(self):
        self.fail()
