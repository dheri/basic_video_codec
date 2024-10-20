from encoder.params import EncoderParameters


class InputParameters:
    def __init__(self, y_only_file, width, height, encoder_parameters: EncoderParameters, frames_to_process=12, yuv_file=None):
        self.yuv_file = yuv_file
        self.y_only_file = y_only_file
        self.width = width
        self.height = height
        self.frames_to_process = frames_to_process
        self.encoder_parameters = encoder_parameters


    def __repr__(self):
        return f"InputParameters(y_only_file={self.y_only_file}, block_size={self.block_size}, search_range={self.search_range}, residual_approx_factor={self.residual_approx_factor})"

