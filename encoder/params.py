import math


class EncoderParameters:
    def __init__(self, block_size, search_range, i_period, quantization_factor):
        validate_qp(block_size, quantization_factor)
        self.block_size = block_size
        self.search_range = search_range
        self.i_period = i_period
        self.quantization_factor = quantization_factor
        self.residual_approx_factor = 0



def validate_qp(i,qp):
    if qp > (math.log2(i) + 7):
        raise ValueError(f" qp [{qp}] > {(math.log2(i) + 7)}")

class EncodedFrame:
    def __init__(self, mv_field, avg_mae, residual_frame_with_mc, quat_dct_coffs_with_mc, reconstructed_frame_with_mc):
        self.mv_field = mv_field
        self.avg_mae = avg_mae
        self.residual_frame_with_mc = residual_frame_with_mc
        self.quat_dct_coffs_with_mc = quat_dct_coffs_with_mc
        self.reconstructed_frame_with_mc = reconstructed_frame_with_mc

class EncodedBlock:
    def __init__(self, block_coords, motion_vector, mae, quantized_dct_coffs, reconstructed_residual_block, reconstructed_block_with_mc ):
        self.block_coords = block_coords
        self.motion_vector = motion_vector
        self.mae = mae
        self.quantized_dct_coffs = quantized_dct_coffs
        self.reconstructed_residual_block = reconstructed_residual_block
        self.reconstructed_block_with_mc = reconstructed_block_with_mc
