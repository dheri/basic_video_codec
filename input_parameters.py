from encoder.params import EncoderConfig


class InputParameters:
    def __init__(self, y_only_file, width, height, encoder_config: EncoderConfig, frames_to_process=12, yuv_file=None):
        self.yuv_file = yuv_file
        self.y_only_file = y_only_file
        self.width = width
        self.height = height
        self.frames_to_process = frames_to_process
        self.encoder_config = encoder_config


