import math

from common import get_logger

logger= get_logger()
class EncoderConfig:
    def __init__(self,ParallelMode, block_size, search_range, I_Period, quantization_factor,
                 nRefFrames=1, fastME=False, fracMeEnabled=False,
                 RCflag=False, targetBR = 0, resolution=(352, 288)):
        self.block_size = block_size
        self.search_range = search_range
        self.quantization_factor = quantization_factor
        self.I_Period = I_Period
        self.residual_approx_factor = 0
        self.nRefFrames = nRefFrames
        self.fastME = fastME
        self.fracMeEnabled = fracMeEnabled
        self.RCflag = RCflag
        self.rc_lookup_table : dict | None = None
        self.targetBR = targetBR
        self.resolution= resolution
        self.frame_rate = 30
        self.ParallelMode = ParallelMode
        self.validate()




    def validate(self):
        if self.quantization_factor > (math.log2(self.block_size) + 7):
            raise ValueError(f" qp [{self.quantization_factor}] > {(math.log2(self.block_size) + 7)}")
        if self.RCflag:
            if self.targetBR ==0:
                raise ValueError("Target Bit Rate is 0 when Rate Control is On")
        if self.fastME:
            self.search_range = -1
        return  self


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
