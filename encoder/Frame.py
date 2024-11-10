import math
from typing import Optional

import numpy as np
from bitarray import bitarray
from skimage.metrics import peak_signal_noise_ratio

from common import get_logger, split_into_blocks, merge_blocks
from encoder.PredictionMode import PredictionMode
from encoder.byte_stream_buffer import BitStreamBuffer
from encoder.entropy_encoder import zigzag_order, rle_encode, exp_golomb_encode, exp_golomb_decode, rle_decode, \
    inverse_zigzag_order
from encoder.params import EncoderConfig
from file_io import write_y_only_frame
from input_parameters import InputParameters

logger = get_logger()

class Frame:
    def __init__(self, curr_frame=None, prev_frame=None, ):
        self.bitstream_buffer : Optional[BitStreamBuffer] = None
        self.prev_frame = prev_frame
        self.curr_frame = curr_frame
        self.prediction_mode: PredictionMode = PredictionMode.INTER_FRAME
        self.prediction_data : Optional[bytearray] = None # should always be byte arrays of unit8
        self.entropy_encoded_prediction_data  : Optional[bitarray] = None
        self.entropy_encoded_DCT_coffs  : Optional[bitarray] = None

        self.residual_frame = None
        self.quantized_dct_residual_frame = None
        self.reconstructed_frame = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode(self, frame_shape, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")
    def generate_prediction_data(self):
        raise NotImplementedError(f"{type(self)} need to be overridden")
    def parse_prediction_data(self, params: InputParameters):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def entropy_encode_prediction_data(self):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def entropy_encode_dct_coffs(self, block_size):
        self.entropy_encoded_DCT_coffs = bitarray()

        blocks = split_into_blocks(self.quantized_dct_residual_frame, block_size)
        for block in blocks:
            zigzag_dct_coffs = zigzag_order(block)
            rle = rle_encode(zigzag_dct_coffs)
            for symbol in rle:
                enc = exp_golomb_encode(symbol)
                self.entropy_encoded_DCT_coffs.extend(enc)

        logger.info(f" entropy_encoded_DCT_coffs  len : {len(self.entropy_encoded_DCT_coffs)}, {len(self.entropy_encoded_DCT_coffs) // 8}")

    def entropy_decode_dct_coffs(self, params: InputParameters):
        block_size = params.encoder_config.block_size
        decoded_blocks = []
        bitstream = self.entropy_encoded_DCT_coffs
        rle_decoded = []

        # Step 1: Decode the Exponential-Golomb encoded symbols
        while bitstream:
            symbol, bitstream = exp_golomb_decode(bitstream)
            if symbol == 0:
                break  # Stop if 0 is encountered, indicating end of meaningful data
            rle_decoded.append(symbol)

        # Step 2: Apply RLE decoding to reconstruct the zigzag order of coefficients
        decoded_coeffs = rle_decode(rle_decoded)  # Assuming rle_decode is defined

        # Step 3: Split coefficients into blocks and apply inverse zigzag order
        num_blocks = len(decoded_coeffs) // (block_size * block_size)
        for i in range(num_blocks):
            block_coeffs = decoded_coeffs[i * (block_size * block_size):(i + 1) * (block_size * block_size)]
            block = inverse_zigzag_order(block_coeffs, block_size)
            decoded_blocks.append(block)

        # Step 4: Reconstruct the frame from the blocks
        self.quantized_dct_residual_frame = merge_blocks(decoded_blocks, block_size, (params.height, params.width) )

        return self.quantized_dct_residual_frame

    def write_metrics_data(self, metrics_csv_writer, frame_index, encoder_config: EncoderConfig):
        psnr = peak_signal_noise_ratio(self.curr_frame, self.reconstructed_frame)
        dct_coffs_extremes = self.get_quat_dct_coffs_extremes()
        logger.info(
            f" {self.prediction_mode:1} {frame_index:2}: i={encoder_config.block_size} r={encoder_config.search_range}, qp={encoder_config.quantization_factor}, mae[{round(self.avg_mae, 2):7.2f}] psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")
        metrics_csv_writer.writerow([frame_index, self.avg_mae, psnr])

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh,residual_yuv_fh, residual_wo_mc_yuv_fh, reconstructed_fh, encoder_config):

        write_y_only_frame(residual_yuv_fh, self.residual_frame)
        write_y_only_frame(residual_wo_mc_yuv_fh, self.residual_frame)
        write_y_only_frame(quant_dct_coff_fh, self.quantized_dct_residual_frame)
        write_y_only_frame(reconstructed_fh, self.reconstructed_frame)

        #encoded_fh.write(self.bitstream_buffer.get_bitstream())

        # mv_fh.write(bytearray(self.prediction_data))



    def construct_frame_metadata_from_bit_stream(self, params : InputParameters, encoded_frame_bytes: bytes):
        """
        reads encoded_frame_bytes and populates prediction_mode, prediction_data, quantized_dct_residual_frame
        :param params:
        :param encoded_frame_bytes: bitstream to read
        :return:
        """
        self.bitstream_buffer = BitStreamBuffer()
        bit_arr = bitarray()
        bit_arr.frombytes(encoded_frame_bytes)
        self.bitstream_buffer.bit_stream = bit_arr

        self.prediction_mode = PredictionMode(self.bitstream_buffer.read_bit()) # pop first
        self.prediction_data = self.bitstream_buffer.read_prediction_data(self.prediction_mode, params)
        self.parse_prediction_data(params)


        self.quantized_dct_residual_frame = self.bitstream_buffer.read_quantized_coeffs(params.height, params.width, params.encoder_config.block_size).astype(np.int16)

        dct_coffs_extremes = self.get_quat_dct_coffs_extremes()
        logger.info(f"quantized_dct shape [{self.quantized_dct_residual_frame.shape}], range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")



    def encoded_frame_data_length(self, params: InputParameters):
        block_size = params.encoder_config.block_size
        num_of_blocks = (params.height // block_size) * (params.width // block_size)
        bits_per_block = 0

        if self.prediction_mode == PredictionMode.INTRA_FRAME:
            bits_per_block +=   1  # 1 bit for inta frame (h/v)
        elif self.prediction_mode == PredictionMode.INTER_FRAME:
            bits_per_block +=   2 * 8 # 2 int8 for mv
        else:
            raise ValueError(f"unexpected prediction_mode: {self.prediction_mode}")
        bits_per_block += (block_size **  2) * (2 * 8) # 16 bits per pixel in block
        bits_per_frame = 1 + (bits_per_block * num_of_blocks)
        bytes_per_frame = math.ceil(bits_per_frame / 8)

        return bytes_per_frame

    def get_quat_dct_coffs_extremes(self):
        # Ensure quat_dct_coffs_with_mc is a numpy array to use numpy's min/max
        if isinstance(self.quantized_dct_residual_frame, np.ndarray):
            min_value = np.min(self.quantized_dct_residual_frame)
            max_value = np.max(self.quantized_dct_residual_frame)
            return [min_value, max_value]
        else:
            raise TypeError(f"{self.quantized_dct_residual_frame} quantized_dct_residual_frame must be a numpy array")

