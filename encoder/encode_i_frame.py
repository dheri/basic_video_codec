import numpy as np

from encoder.dct import *

import numpy as np

def intra_predict_block(curr_block, reconstructed_frame, x, y, block_size):
    """Predict the block using horizontal and vertical intra prediction, and choose the best mode based on MAE."""
    horizontal_pred = horizontal_intra_prediction(reconstructed_frame, x, y, block_size)
    vertical_pred = vertical_intra_prediction(reconstructed_frame, x, y, block_size)

    mae_horizontal = np.mean(np.abs(curr_block - horizontal_pred))
    mae_vertical = np.mean(np.abs(curr_block - vertical_pred))

    if mae_horizontal < mae_vertical:
        return horizontal_pred, 0, mae_horizontal  # 0 for horizontal mode
    else:
        return vertical_pred, 1, mae_vertical  # 1 for vertical mode


def horizontal_intra_prediction(reconstructed_frame, x, y, block_size):
    """Perform horizontal intra prediction using the left border samples."""
    if x > 0:
        left_samples = reconstructed_frame[y:y + block_size, x - 1]
        return np.tile(left_samples, (block_size, 1))
    else:
        return np.full((block_size, block_size), 128)  # Use 128 for border


def vertical_intra_prediction(reconstructed_frame, x, y, block_size):
    """Perform vertical intra prediction using the top border samples."""
    if y > 0:
        top_samples = reconstructed_frame[y - 1, x:x + block_size]
        return np.tile(top_samples, (block_size, 1)).T
    else:
        return np.full((block_size, block_size), 128)  # Use 128 for border


def process_block(curr_block, reconstructed_frame, x, y, block_size, quantization_factor):
    """Process a block, apply intra prediction, DCT, quantization, and reconstruction."""
    predicted_block, mode, mae = intra_predict_block(curr_block, reconstructed_frame, x, y, block_size)

    # Compute the residual
    residual_block = curr_block.astype(np.int16) - predicted_block.astype(np.int16)

    # Apply DCT
    dct_residual_block = apply_dct_2d(residual_block)

    # Quantization
    Q = generate_quantization_matrix(block_size, quantization_factor)
    quantized_dct_residual_block = quantize_block(dct_residual_block, Q)

    # Inverse quantization and IDCT
    dequantized_dct_residual_block = rescale_block(quantized_dct_residual_block, Q)
    reconstructed_residual_block = apply_idct_2d(dequantized_dct_residual_block)

    # Reconstruct the block
    reconstructed_block = np.round(predicted_block + reconstructed_residual_block).astype(np.uint8)

    return mode, mae, reconstructed_block, quantized_dct_residual_block


def encode_i_frame(curr_frame, encoder_params):
    """Encode I-frame using intra prediction and quantization."""
    block_size = encoder_params.block_size
    height, width = curr_frame.shape

    mv_field = {}
    mae_of_blocks = 0
    intra_modes = []  # To store the intra prediction modes (0 for horizontal, 1 for vertical)
    reconstructed_frame = np.zeros_like(curr_frame)
    quantized_dct_residual_frame = np.zeros_like(curr_frame, dtype=np.int16)

    # Loop through each block in the frame
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = curr_frame[y:y + block_size, x:x + block_size]

            # Process the block
            mode, mae, reconstructed_block, quantized_dct_residual_block = process_block(
                curr_block, reconstructed_frame, x, y, block_size, encoder_params.quantization_factor
            )

            # Store intra mode and update MAE
            intra_modes.append(mode)
            mae_of_blocks += mae

            # Update reconstructed frame and quantized residuals
            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block
            quantized_dct_residual_frame[y:y + block_size, x:x + block_size] = quantized_dct_residual_block

    avg_mae = mae_of_blocks / ((height // block_size) * (width // block_size))

    return {
        'intra_modes': intra_modes,
        'reconstructed_frame': reconstructed_frame,
        'avg_mae': avg_mae,
        'quantized_dct_residual_frame': quantized_dct_residual_frame
    }

