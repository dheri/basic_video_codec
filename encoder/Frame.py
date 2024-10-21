from enum import Enum
from typing import Optional

import numpy as np

from encoder.byte_stream_buffer import BitStreamBuffer
from encoder.params import EncoderConfig


class PredictionMode(Enum):
    INTER_FRAME = 0  # P-frame
    INTRA_FRAME = 1  # I-frame


class FrameHeader:
    """Class to handle frame header information such as frame type and size."""
    def __init__(self, frame_type: PredictionMode, size: int):

        self.size = size



class Frame:
    def __init__(self, curr_frame=None, prev_frame=None, ):
        self.bitstream_buffer : Optional[BitStreamBuffer] = None
        self.prev_frame = prev_frame
        self.curr_frame = curr_frame
        self.prediction_mode: PredictionMode = PredictionMode.INTER_FRAME
        self.residual_frame = None
        self.quantized_dct_residual_frame = None
        self.reconstructed_frame = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_metrics_data(self, metrics_csv_writer, frame_index : int, encoder_config : EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh):
        pass

    def pre_entropy_encoded_frame_bit_stream(self) -> BitStreamBuffer:
        self.bitstream_buffer = BitStreamBuffer()
        self.bitstream_buffer.write_bit(self.prediction_mode.value)
        if self.quantized_dct_residual_frame is not None:
            self.bitstream_buffer.write_quantized_coeffs(self.quantized_dct_residual_frame)
        else:
            raise ValueError("Illegal operation: quantized_dct_residual_frame is None")


        return self.bitstream_buffer

    def read_entropy_encoded_frame_bit_stream(self) -> BitStreamBuffer:
        pass

    def encoded_frame_data_length(self):
        length = 0
        # for  self.prediction_mode.value
        length += 1

        # for  self.prediction_mode.value
        length += self.quantized_dct_residual_frame.size * 16

        # padding
        length += 8 - (length % 8)
        return length

    def get_quat_dct_coffs_extremes(self):
        # Ensure quat_dct_coffs_with_mc is a numpy array to use numpy's min/max
        if isinstance(self.quantized_dct_residual_frame, np.ndarray):
            min_value = np.min(self.quantized_dct_residual_frame)
            max_value = np.max(self.quantized_dct_residual_frame)
            return [min_value, max_value]
        else:
            raise TypeError("quantized_dct_residual_frame must be a numpy array")

