import math

from common import get_logger

logger= get_logger()
class EncoderConfig:
    def __init__(self, block_size, search_range, I_Period, quantization_factor, nRefFrames=1, fastME=False, FMEEnable=False):
        self.block_size = block_size
        self.search_range = search_range
        self.quantization_factor = quantization_factor
        self.I_Period = I_Period
        self.residual_approx_factor = 0
        self.nRefFrames = nRefFrames
        self.fastME = fastME
        self.FMEEnable = FMEEnable
        self.validate()




    def validate(self):
        if self.quantization_factor > (math.log2(self.block_size) + 7):
            raise ValueError(f" qp [{self.quantization_factor}] > {(math.log2(self.block_size) + 7)}")
        if self.fastME:
            self.search_range = -1


class EncodedBlock:
    def __init__(self, block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                 reconstructed_block_with_mc, comparisons):
        self.block_coords = block_coords
        self.mae = mae
        self.quantized_dct_coffs = quantized_dct_coffs
        self.reconstructed_residual_block = reconstructed_residual_block
        self.residual_block_wo_mc = residual_block_wo_mc
        self.reconstructed_block = reconstructed_block_with_mc
        self.mae_comparisons_to_encode = comparisons


class EncodedPBlock(EncodedBlock):
    def __init__(self, block_coords, motion_vector, mae, quantized_dct_coffs, reconstructed_residual_block,
                 residual_block_wo_mc, reconstructed_block_with_mc, comparisons):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                         reconstructed_block_with_mc, comparisons)
        self.motion_vector = motion_vector


class EncodedIBlock(EncodedBlock):
    def __init__(self, block_coords, mode, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                 reconstructed_block):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc,
                         reconstructed_block, 2)# 2 comparisons regardless of search range
        self.mode = mode
