from unittest import TestCase

from encoder.PredictionMode import PredictionMode
from encoder.byte_stream_buffer import BitStreamBuffer
import numpy as np

class TestBitStreamBuffer(TestCase):
    def test_read_bits_2(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_int8(42)
        bit_stream_buff.write_int8(-5)

        bit_stream_buff.flush()

        print(bit_stream_buff)

        self.assertEqual(1,bit_stream_buff.read_bit())
        self.assertEqual(42,bit_stream_buff.read_int8())
        self.assertEqual(-5, bit_stream_buff.read_int8())
        for i in range (2):
            self.assertEqual(0,bit_stream_buff.read_bit())

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
        bit_stream_buff.write_int16(-42)

        print(bit_stream_buff)

        for i in range (8):
            self.assertEqual(0,bit_stream_buff.read_bit())

        self.assertEqual(1,bit_stream_buff.read_bit())
        self.assertEqual(0,bit_stream_buff.read_bit())

        for i in range (16):
            self.assertEqual(1,bit_stream_buff.read_bit() )


        for i in range (6):
            self.assertEqual(0,bit_stream_buff.read_bit())
        self.assertEqual(-42,bit_stream_buff.read_int16())

    def test_read_prediction_data_intra(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_prediction_data(PredictionMode.INTRA_FRAME, [0,1])
        bit_stream_buff.write_quantized_coeffs( np.array([[ -41], [9]]), 1)

        bit_stream_buff.flush()
        print(bit_stream_buff)

        self.assertEqual(1, bit_stream_buff.read_bit())
        self.assertEqual(0, bit_stream_buff.read_bit())
        self.assertEqual(1, bit_stream_buff.read_bit())
        self.assertEqual(-41, bit_stream_buff.read_int16())
        self.assertEqual(9, bit_stream_buff.read_int16())


    def test_read_prediction_data_inter(self):
        bit_stream_buff = BitStreamBuffer()
        bit_stream_buff.write_bit(1)
        bit_stream_buff.write_prediction_data(PredictionMode.INTER_FRAME, [-2, 1, -5, 6])
        bit_stream_buff.write_quantized_coeffs( np.array([[ -41], [9]]), 1)

        bit_stream_buff.flush()
        print(bit_stream_buff)

        self.assertEqual(1, bit_stream_buff.read_bit())
        self.assertEqual(-2, bit_stream_buff.read_int8())
        self.assertEqual(1, bit_stream_buff.read_int8())
        self.assertEqual(-5, bit_stream_buff.read_int8())
        self.assertEqual(6, bit_stream_buff.read_int8())

        self.assertEqual(-41, bit_stream_buff.read_int16())
        self.assertEqual(9, bit_stream_buff.read_int16())
