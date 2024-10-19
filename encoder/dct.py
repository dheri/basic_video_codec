import numpy as np
from scipy.fftpack import dct, idct  # For 2D DCT and inverse DCT

def apply_dct_2d(block):
    """
    Applies 2D DCT to a block.
    :param block: (i x i) residual block
    :return: DCT-transformed block
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct_2d(block):
    """
    Applies inverse 2D DCT to a block.
    :param block: (i x i) transformed block
    :return: Reconstructed block using inverse DCT
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, q_factor):
    """
    Applies quantization by rounding DCT coefficients.
    :param block: DCT-transformed block
    :param q_factor: Quantization factor
    :return: Quantized DCT coefficients
    """
    return np.round(block / q_factor) * q_factor

def inverse_quantize(block, q_factor):
    """
    Reverses quantization of DCT coefficients.
    :param block: Quantized DCT-transformed block
    :param q_factor: Quantization factor
    :return: Dequantized DCT coefficients
    """
    return block * q_factor