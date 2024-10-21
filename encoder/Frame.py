from enum import Enum

from encoder.encode_i_frame import encode_i_frame
from encoder.params import EncoderConfig
from file_io import FileIOHelper


class PredictionMode(Enum):
    INTER_FRAME = 0  # P-frame
    INTRA_FRAME = 1  # I-frame


class FrameHeader:
    """Class to handle frame header information such as frame type and size."""
    def __init__(self, frame_type: PredictionMode, size: int):
        self.frame_type = frame_type
        self.size = size

    def __str__(self):
        return f"FrameHeader(Type: {self.frame_type.name}, Size: {self.size})"


class Frame:
    def __init__(self, curr_frame, prev_frame=None, ):
        self.prev_frame = prev_frame
        self.curr_frame = curr_frame
        self.mode = PredictionMode.INTER_FRAME if prev_frame is not None else PredictionMode.INTRA_FRAME
        self.header = None

    def encode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_metrics_data(self, metrics_csv_writer, frame_index : int, encoder_config : EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh,residual_yuv_fh , reconstructed_fh):
        pass

    def read_from_file(self):
        pass

