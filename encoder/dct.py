import numpy as np
from scipy.fftpack import dct, idct

from common import get_logger

logger = get_logger()


def apply_dct_2d(block):
    """Applies 2D DCT to a block using separable 1D DCT."""
    block = block.astype(np.float32)
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct_2d(block):
    """Applies 2D Inverse DCT to a block."""
    block = block.astype(np.float32)
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


