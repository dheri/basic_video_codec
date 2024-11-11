import math

import numpy as np


class EncoderConfig:
    def __init__(self, block_size, search_range, I_Period , quantization_factor):
        validate_qp(block_size, quantization_factor)
        self.block_size = block_size
        self.search_range = search_range
        self.quantization_factor = quantization_factor
        self.I_Period = I_Period
        self.residual_approx_factor = 0



def validate_qp(i,qp):
    if qp > (math.log2(i) + 7):
        raise ValueError(f" qp [{qp}] > {(math.log2(i) + 7)}")


class EncodedBlock:
    def __init__(self, block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc, reconstructed_block_with_mc ):
        self.block_coords = block_coords
        self.mae = mae
        self.quantized_dct_coffs = quantized_dct_coffs
        self.reconstructed_residual_block = reconstructed_residual_block
        self.residual_block_wo_mc = residual_block_wo_mc
        self.reconstructed_block = reconstructed_block_with_mc


class EncodedPBlock(EncodedBlock):
    def __init__(self, block_coords, motion_vector, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc, reconstructed_block_with_mc ):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc, reconstructed_block_with_mc )
        self.motion_vector = motion_vector


class EncodedIBlock(EncodedBlock):
    def __init__(self, block_coords, mode, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc, reconstructed_block ):
        super().__init__(block_coords, mae, quantized_dct_coffs, reconstructed_residual_block, residual_block_wo_mc, reconstructed_block )
        self.mode = mode

