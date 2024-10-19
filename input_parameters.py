class EncoderParameters:
    def __init__(self, block_size, search_range, i_period, qp, residual_approx_factor=0 ):
        self.block_size = block_size
        self.search_range = search_range
        self.i_period = i_period
        self.qp = qp
        self.residual_approx_factor = residual_approx_factor

class InputParameters:
    def __init__(self, y_only_file, width, height, frames_to_process, encoder_parameters: EncoderParameters, yuv_file=None):
        self.yuv_file = yuv_file
        self.y_only_file = y_only_file
        self.width = width
        self.height = height
        self.frames_to_process = frames_to_process
        self.encoder_parameters = encoder_parameters


    def __repr__(self):
        return f"InputParameters(y_only_file={self.y_only_file}, block_size={self.block_size}, search_range={self.search_range}, residual_approx_factor={self.residual_approx_factor})"

