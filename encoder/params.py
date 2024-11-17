import math


class EncoderConfig:
    def __init__(self, block_size, search_range, I_Period, quantization_factor, 
                 nRefFrames=1, VBSEnable=False, lambda_value=0.85):
        validate_qp(block_size, quantization_factor)
        self.block_size = block_size
        self.search_range = search_range
        self.quantization_factor = quantization_factor
        self.I_Period = I_Period
        self.residual_approx_factor = 0
        self.nRefFrames = nRefFrames
        self.VBSEnable = VBSEnable 
        self.lambda_value = lambda_value 
        self.QP_range = (0, 51)  
   

    def validate_lambda_and_qp(self):
        if not (0 <= self.lambda_value <= 10):  
            raise ValueError(f"Invalid lambda value: {self.lambda_value}. Must be between 0 and 10.")
        if not (self.QP_range[0] <= self.quantization_factor <= self.QP_range[1]):
            raise ValueError(f"Quantization factor {self.quantization_factor} out of range {self.QP_range}.")


def validate_qp(i, qp):
    if qp > (math.log2(i) + 7):
        raise ValueError(f" qp [{qp}] > {(math.log2(i) + 7)}")


class EncodedBlock:
    def __init__(self, block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                 reconstructed_block_with_mc):
        self.block_coords = block_coords
        self.mae = mae
        self.quantized_dct_coffs = quantized_dct_coffs
        self.reconstructed_residual_block = reconstructed_residual_block
        self.residual_block_wo_mc = residual_block_wo_mc
        self.reconstructed_block = reconstructed_block_with_mc


class EncodedPBlock(EncodedBlock):
    def __init__(self, block_coords, motion_vector, mae, quantized_dct_coffs, reconstructed_residual_block,
                 residual_block_wo_mc, reconstructed_block_with_mc, split=False, sub_blocks=None):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                         reconstructed_block_with_mc)
        self.motion_vector = motion_vector
        self.split = split  # Indicates whether the block was split (VBS)
        self.sub_blocks = sub_blocks if sub_blocks is not None else []  # Sub-block information


class EncodedIBlock(EncodedBlock):
    def __init__(self, block_coords, mode, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                 reconstructed_block, split=False, sub_blocks=None):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                         reconstructed_block)
        self.mode = mode
        self.split = split  # Indicates whether the block was split (VBS)
        self.sub_blocks = sub_blocks if sub_blocks is not None else []  # Sub-block information
