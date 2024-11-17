import numpy as np
from bitarray import bitarray

from common import get_logger
from encoder.Frame import Frame, apply_dct_and_quantization, reconstruct_block
from encoder.PredictionMode import PredictionMode
from encoder.dct import generate_quantization_matrix, rescale_block, apply_idct_2d
from encoder.entropy_encoder import exp_golomb_encode, exp_golomb_decode
from encoder.params import EncoderConfig, EncodedIBlock
from encoder.block_predictor import *

logger = get_logger()


class IFrame(Frame):
    def __init__(self, curr_frame=None):
        super().__init__(curr_frame)
        self.prediction_mode = PredictionMode.INTRA_FRAME
        self.intra_modes = None
        self.split_flags = []  # For storing split information for each block

    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        curr_frame = self.curr_frame
        block_size = encoder_config.block_size
        height, width = curr_frame.shape
        lambda_value = 0.1 * encoder_config.quantization_factor  # Example lambda-QP relation

        mae_of_blocks = 0
        intra_modes = []  # To store the intra prediction modes (0 for horizontal, 1 for vertical)
        split_flags = []  # To store the split flag for each block
        reconstructed_frame = np.zeros_like(curr_frame)
        residual_w_mc_frame = np.zeros_like(curr_frame)
        quantized_dct_residual_frame = np.zeros_like(curr_frame, dtype=np.int16)

        # Loop through each block in the frame
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                curr_block = curr_frame[y:y + block_size, x:x + block_size]

                if encoder_config.VBSEnable:
                    # Use VBS logic if enabled
                    encoded_data = process_block_vbs(
                        curr_block, reconstructed_frame, x, y, block_size,
                        encoder_config.quantization_factor, lambda_value, apply_dct_and_quantization
                    )

                    split_flags.append(encoded_data["split"])  # Store the split flag

                    if encoded_data["split"]:
                        # Process sub-blocks
                        for sub_block in encoded_data["sub_blocks"]:
                            sub_x, sub_y = sub_block["coords"]
                            sub_size = block_size // 2
                            intra_modes.append(sub_block["mode"])
                            mae_of_blocks += sub_block["mae"]
                            reconstructed_frame[sub_y:sub_y + sub_size, sub_x:sub_x + sub_size] = sub_block["predicted_block"]
                            quantized_dct_residual_frame[sub_y:sub_y + sub_size, sub_x:sub_x + sub_size] = sub_block[
                                "quantized_dct_coffs"]
                            residual_w_mc_frame[sub_y:sub_y + sub_size, sub_x:sub_x + sub_size] = sub_block["residual"]
                    else:
                        # Process as a single block
                        intra_modes.append(encoded_data["mode"])
                        mae_of_blocks += encoded_data["mae"]
                        reconstructed_frame[y:y + block_size, x:x + block_size] = encoded_data["predicted_block"]
                        quantized_dct_residual_frame[y:y + block_size, x:x + block_size] = encoded_data[
                            "quantized_dct_coffs"]
                        residual_w_mc_frame[y:y + block_size, x:x + block_size] = encoded_data["residual"]

                else:
                    # Standard processing without VBS
                    encoded_block = process_block(
                        curr_block, reconstructed_frame, x, y, block_size, encoder_config.quantization_factor
                    )
                    intra_modes.append(encoded_block.mode)
                    mae_of_blocks += encoded_block.mae
                    reconstructed_frame[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                    quantized_dct_residual_frame[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs
                    residual_w_mc_frame[y:y + block_size, x:x + block_size] = encoded_block.residual_block_wo_mc

        avg_mae = mae_of_blocks / ((height // block_size) * (width // block_size))
        self.reconstructed_frame = reconstructed_frame
        self.quantized_dct_residual_frame = quantized_dct_residual_frame
        self.intra_modes = intra_modes
        self.split_flags = split_flags
        self.avg_mae = avg_mae
        self.residual_frame = residual_w_mc_frame
        self.residual_wo_mc_frame = residual_w_mc_frame
        

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        block_size = encoder_config.block_size
        height, width = frame_shape
        reconstructed_frame = np.zeros((height, width), dtype=np.uint8)

        # Decode split flags and sub-blocks
        Q = generate_quantization_matrix(block_size, encoder_config.quantization_factor)

        # Iterate over blocks to reconstruct the frame
        split_index = 0
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if encoder_config.VBSEnable and self.split_flags[split_index]:
                    # Decode sub-blocks
                    sub_block_size = block_size // 2
                    for sub_y in range(0, block_size, sub_block_size):
                        for sub_x in range(0, block_size, sub_block_size):
                            dct_coffs_block = self.quantized_dct_residual_frame[
                                              y + sub_y:y + sub_y + sub_block_size,
                                              x + sub_x:x + sub_x + sub_block_size
                                              ]
                            rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)
                            idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)
                            predicted_b = find_intra_predict_block(self.intra_modes.pop(0), reconstructed_frame,
                                                                   x + sub_x, y + sub_y, sub_block_size)
                            decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)
                            decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)
                            reconstructed_frame[y + sub_y:y + sub_y + sub_block_size,
                                                x + sub_x:x + sub_x + sub_block_size] = decoded_block
                else:
                    # Decode single block
                    dct_coffs_block = self.quantized_dct_residual_frame[y:y + block_size, x:x + block_size]
                    rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)
                    idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)
                    predicted_b = find_intra_predict_block(self.intra_modes.pop(0), reconstructed_frame,
                                                           x, y, block_size)
                    decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)
                    decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)
                    reconstructed_frame[y:y + block_size, x:x + block_size] = decoded_block

                split_index += 1

        self.curr_frame = reconstructed_frame
        return reconstructed_frame  # This should be the reconstructed fra
    def entropy_encode_prediction_data(self):
        self.entropy_encoded_prediction_data = bitarray()
        # logger.info(self.intra_modes)
        for m in self.intra_modes:
            enc = exp_golomb_encode(m)
            self.entropy_encoded_prediction_data.extend(enc)
        # logger.info(f" entropy_encoded_prediction_data  len : {len(self.entropy_encoded_prediction_data)}, {len(self.entropy_encoded_prediction_data) // 8}")
        # logger.info(self.entropy_encoded_prediction_data)

    def entropy_decode_prediction_data(self, enc, params=None):
        decoded_modes = []
        bitstream = bitarray()  # Ensure `enc` is a bitarray
        bitstream.frombytes(enc)

        # Decode each encoded mode
        while bitstream:
            try:
                decoded_value, bitstream = exp_golomb_decode(bitstream)
                decoded_modes.append(decoded_value)
            except ValueError as e:
                # Handle the case where bitstream is exhausted or an error occurs
                logger.error(f"Decoding error: {e}")
                break
        self.intra_modes = decoded_modes
        return decoded_modes


def find_intra_predict_block(prediction_mode, reconstructed_frame, x, y, block_size):
    """Find the predicted block based on the specified prediction mode."""
    if prediction_mode == 0:  # Horizontal prediction mode
        return horizontal_intra_prediction(reconstructed_frame, x, y, block_size)
    elif prediction_mode == 1:  # Vertical prediction mode
        return vertical_intra_prediction(reconstructed_frame, x, y, block_size)
    else:
        raise ValueError(f"Invalid prediction mode [{prediction_mode}]: must be 0 (horizontal) or 1 (vertical).")


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


def process_block(curr_block, reconstructed_frame, x, y, block_size, quantization_factor) -> EncodedIBlock:
    """Process a block, apply intra prediction, DCT, quantization, and reconstruction."""
    predicted_block, mode, mae = intra_predict_block(curr_block, reconstructed_frame, x, y, block_size)

    # Compute the residual
    # residual_block = curr_block.astype(np.int16) - predicted_block.astype(np.int16)
    residual_block = np.subtract(curr_block.astype(np.int16), predicted_block.astype(np.int16))

    # Apply DCT
    quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block, block_size, quantization_factor)

    clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q,
                                                                         predicted_block)

    return EncodedIBlock((x, y), mode, mae, quantized_dct_coffs, idct_residual_block, residual_block,
                         clipped_reconstructed_block)

    # return mode, mae, clipped_reconstructed_block, quantized_dct_coffs, residual_block
