from unittest import TestCase

from encoder.PredictionMode import PredictionMode
from encoder.byte_stream_buffer import BitStreamBuffer
import numpy as np

class TestBitStreamBuffer(TestCase):
    def test_read_bits_1(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_bit(0)
        bit_stream_buff.write_bit(1)

        bit_stream_buff.flush()

        print(bit_stream_buff)
        self.assertEqual(1,bit_stream_buff.read_bit())
        self.assertEqual(0,bit_stream_buff.read_bit())
        self.assertEqual(1,bit_stream_buff.read_bit())
        for i in range (5):
            self.assertEqual(0,bit_stream_buff.read_bit())

    def test_read_bits(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_int8(0)
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_bit(0)
        bit_stream_buff.write_int16(-1)

        bit_stream_buff.flush()

        print(bit_stream_buff)

        for i in range (8):
            self.assertEqual(0,bit_stream_buff.read_bit())

        self.assertEqual(1,bit_stream_buff.read_bit())
        self.assertEqual(0,bit_stream_buff.read_bit())

        for i in range (16):
            self.assertEqual(1,bit_stream_buff.read_bit() )


        for i in range (6):
            self.assertEqual(0,bit_stream_buff.read_bit())

    def test_read_prediction_data(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_prediction_data(PredictionMode.INTRA_FRAME, [0,0])
        bit_stream_buff.write_quantized_coeffs( np.array([[ -1], [0]]))
        print(bit_stream_buff)
        self.assertEqual(1, bit_stream_buff.read_bit())
        self.assertEqual(0, bit_stream_buff.read_bit())
        self.assertEqual(0, bit_stream_buff.read_bit())

