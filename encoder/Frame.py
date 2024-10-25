import math
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from common import get_logger
from encoder.PredictionMode import PredictionMode
from encoder.byte_stream_buffer import BitStreamBuffer
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
        self.prediction_data : Optional[list] = None
        self.residual_frame = None
        self.quantized_dct_residual_frame = None
        self.reconstructed_frame = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode(self,frame_size, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")
    def fix_mv(self):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_metrics_data(self, metrics_csv_writer, frame_index, encoder_config: EncoderConfig):
        psnr = peak_signal_noise_ratio(self.curr_frame, self.reconstructed_frame)
        dct_coffs_extremes = self.get_quat_dct_coffs_extremes()
        logger.info(
            f" {self.prediction_mode:1} {frame_index:2}: i={encoder_config.block_size} r={encoder_config.search_range}, qp={encoder_config.quantization_factor}, mae[{round(self.avg_mae, 2):7.2f}] psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")
        metrics_csv_writer.writerow([frame_index, self.avg_mae, psnr])

    def write_encoded_to_file(self, encoded_fh, mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh, encoder_config):

        write_y_only_frame(residual_yuv_fh, self.residual_frame)
        write_y_only_frame(quant_dct_coff_fh, self.quantized_dct_residual_frame)
        write_y_only_frame(reconstructed_fh, self.reconstructed_frame)

        self.generate_pre_entropy_encoded_frame_bit_stream( encoder_config)
        encoded_fh.write(self.bitstream_buffer.get_bitstream())

        mv_fh.write(bytearray(self.prediction_data))

    def generate_pre_entropy_encoded_frame_bit_stream(self, encoder_config : EncoderConfig) -> BitStreamBuffer:
        self.bitstream_buffer = BitStreamBuffer()
        self.bitstream_buffer.write_bit(self.prediction_mode.value)
        self.bitstream_buffer.write_prediction_data(self.prediction_mode, self.prediction_data)
        self.bitstream_buffer.write_quantized_coeffs(self.quantized_dct_residual_frame, encoder_config.block_size)
        self.bitstream_buffer.flush()
        # logger.info(f"len after flush {len(self.bitstream_buffer.byte_stream)}")
        return self.bitstream_buffer

    def construct_frame_metadata_from_bit_stream(self, params : InputParameters, encoded_fh):
        block_size = params.encoder_config.block_size
        num_of_blocks = (params.height // block_size) * (params.width // block_size)


        bytes_per_frame = self.encoded_frame_data_length(params)

        logger.info(f"reading {bytes_per_frame} bytes for {self.prediction_mode}")

        encoded_frame_bytes = encoded_fh.read(bytes_per_frame)
        self.bitstream_buffer = BitStreamBuffer()
        self.bitstream_buffer.byte_stream = bytearray(encoded_frame_bytes)
        self.prediction_data = self.bitstream_buffer.read_prediction_data(self.prediction_mode, num_of_blocks)
        self.quantized_dct_residual_frame = self.bitstream_buffer.read_quantized_coeffs(params.height, params.width)
        logger.info(f"  construct_frame_metadata dct_coffs_extremes {self.get_quat_dct_coffs_extremes()}")


        # self.fix_mv(num_of_blocks)


    def encoded_frame_data_length(self, params: InputParameters):
        block_size = params.encoder_config.block_size
        num_of_blocks = (params.height // block_size) * (params.width // block_size)
        bits_per_block = 0

        if self.prediction_mode == PredictionMode.INTRA_FRAME:
            bits_per_block +=   1  # 1 bit for inta frame (h/v)
        elif self.prediction_mode == PredictionMode.INTER_FRAME:
            bits_per_block +=   num_of_blocks * 2 * 8 # 2 int8 for mv
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
            raise TypeError("quantized_dct_residual_frame must be a numpy array")

