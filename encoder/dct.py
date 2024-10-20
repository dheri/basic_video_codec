import math

import numpy as np
from scipy.fftpack import dct, idct

from common import get_logger
from encoder.params import validate_qp

logger = get_logger()

def apply_dct_2d(block):
    """Applies 2D DCT to a block using separable 1D DCT."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct_2d(block):
    """Applies 2D Inverse DCT to a block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def generate_quantization_matrix(i, qp):
    """Generates the quantization matrix Q for a given block size i and quantization parameter QP."""
    Q = np.zeros((i, i), dtype=np.uint16)
    for x in range(i):
        for y in range(i):
            if (x + y) < (i - 1):
                Q[x, y] = 2 ** qp
            elif (x + y) == (i - 1):
                Q[x, y] = 2 ** (qp + 1)
            else:
                Q[x, y] = 2 ** (qp + 2)
    return Q

def quantize_block(dct_block, Q):
    """Quantizes a block by dividing by Q and rounding."""
    return np.round(dct_block / Q)

def rescale_block(quantized_block, Q):
    """Rescales the quantized block by multiplying by Q."""
    return quantized_block * Q


def transform_quantize_rescale_inverse(block, qp):
    """Applies the full pipeline: DCT -> Quantization -> Rescale -> IDCT."""
    i = block.shape[0]  # Block size (e.g., 8 for an 8x8 block)

    # logger.info(f'block: \n{np.ceil(block)}')
    validate_qp(i, qp)

    # Step 1: Apply 2D DCT to the block
    dct_coffs = apply_dct_2d(block)
    # logger.info(f'dct_block: \n{np.ceil(dct_block)}')

    # Step 2: Generate the quantization matrix based on block size and QP
    Q = generate_quantization_matrix(i, qp)
    # logger.info(f'Q: \n{np.ceil(Q)}')

    # Step 3: Quantize the DCT coefficients
    quantized_dct_coffs = quantize_block(dct_coffs, Q)
    # logger.info(f'quantized_block: \n{np.ceil(quantized_block)}')

    # Step 4: Rescale the quantized block by multiplying by Q
    rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
    # logger.info(f'rescaled_block: \n{np.ceil(rescaled_block)}')

    # Step 5: Apply Inverse DCT to reconstruct the block
    reconstructed_block = apply_idct_2d(rescaled_dct_coffs)
    # logger.info(f'reconstructed_block: \n{np.ceil(reconstructed_block)}')

    return reconstructed_block
