class InputParameters:
    def __init__(self, y_only_file, width, height, block_size, search_range, residual_approx_factor, frames_to_process, yuv_file=None):
        self.yuv_file = yuv_file
        self.y_only_file = y_only_file
        self.width = width
        self.height = height
        self.block_size = block_size
        self.search_range = search_range
        self.residual_approx_factor = residual_approx_factor
        self.frames_to_process = frames_to_process


    def __repr__(self):
        return f"InputParameters(y_only_file={self.y_only_file}, block_size={self.block_size}, search_range={self.search_range}, residual_approx_factor={self.residual_approx_factor})"
