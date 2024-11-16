from collections import OrderedDict

import numpy as np
from bitarray import bitarray

from common import get_logger, generate_residual_block, find_mv_predicted_block
from encoder.Frame import Frame, apply_dct_and_quantization, reconstruct_block
from encoder.PredictionMode import PredictionMode
from encoder.block_predictor import predict_block
from encoder.dct import generate_quantization_matrix, rescale_block, apply_idct_2d
from encoder.entropy_encoder import exp_golomb_encode, exp_golomb_decode
from encoder.params import EncoderConfig, EncodedPBlock
from input_parameters import InputParameters

logger = get_logger()


class PFrame(Frame):
    def __init__(self, curr_frame=None, reference_frames=None):
        super().__init__(curr_frame, reference_frames)
        self.prediction_mode = PredictionMode.INTER_FRAME
        self.mv_field = None
        self.avg_mae = None

    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        block_size = encoder_config.block_size
        search_range = encoder_config.search_range
        quantization_factor = encoder_config.quantization_factor

        height, width = self.curr_frame.shape
        num_of_blocks = (height // block_size) * (width // block_size)
        mv_field = {}
        mae_of_blocks = 0

        # Initialize output frames
        reconstructed_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.uint8)
        residual_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        residual_frame_wo_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        quat_dct_coffs_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int16)

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                encoded_block =self.process_block( x, y, block_size, search_range, quantization_factor, width, height, mv_field)
                block_cords = encoded_block.block_coords
                x, y = block_cords

                # Update frames with the encoded block data
                reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
                residual_frame_wo_mc[y:y + block_size, x:x + block_size] = encoded_block.residual_block_wo_mc
                quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

                mae_of_blocks += encoded_block.mae

        avg_mae = mae_of_blocks / num_of_blocks

        sorted_mv_field = OrderedDict(sorted(mv_field.items(), key=lambda item: (item[0][1], item[0][0])))
        self.mv_field = sorted_mv_field  # Populate the motion vector field

        self.avg_mae = avg_mae
        self.residual_frame = residual_frame_with_mc
        self.residual_wo_mc_frame = residual_frame_wo_mc
        self.quantized_dct_residual_frame = quat_dct_coffs_frame_with_mc
        self.reconstructed_frame = reconstructed_frame_with_mc
        return self

    def process_block(self, x, y, block_size, search_range, quantization_factor, width, height,
                      mv_field) -> EncodedPBlock:
        curr_block = self.curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)
        # consider latest ref frame for no mv
        prev_block = self.reference_frames[0][y:y + block_size, x:x + block_size].astype(np.int16)

        # Get motion vector and MAE
        motion_vector, best_match_mae = self.get_motion_vector(curr_block, x, y, block_size, search_range)
        mv_field[(x, y)] = motion_vector

        # Generate residual and predicted block
        predicted_block_with_mc, residual_block_with_mc = generate_residual_block(curr_block, self.reference_frames,
                                                                                  motion_vector, x, y, block_size)
        residual_block_wo_mc = np.subtract(curr_block, prev_block)
        # Apply DCT and quantization
        quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

        # Reconstruct the block using the predicted and inverse DCT
        clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q,
                                                                             predicted_block_with_mc)
        check_index_out_of_bounds(x, y, motion_vector,width,height, block_size)

        return EncodedPBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block,
                             residual_block_wo_mc, clipped_reconstructed_block)

    def get_motion_vector(self, curr_block, x, y, block_size, search_range):
        mv, best_match_mae, best_match_block = predict_block(curr_block, (x,y), self.reference_frames,block_size, search_range)
        return mv, best_match_mae

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        return construct_frame_from_dct_and_mv(self.quantized_dct_residual_frame, self.reference_frames, self.mv_field,
                                               encoder_config)

    def entropy_encode_prediction_data(self, encoder_config:EncoderConfig):
        self.entropy_encoded_prediction_data = bitarray()
        for key, mv in self.mv_field.items():
            enc_mv_x = exp_golomb_encode(mv[0])
            self.entropy_encoded_prediction_data.extend(enc_mv_x)
            enc_mv_y = exp_golomb_encode(mv[1])
            self.entropy_encoded_prediction_data.extend(enc_mv_y)
            if encoder_config.nRefFrames>1:
                enc_mv_ref_frame_idx = exp_golomb_encode(mv[2])
                self.entropy_encoded_prediction_data.extend(enc_mv_ref_frame_idx)
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()} {enc_mv_ref_frame_idx.to01()}]")
            else:
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()}]")

        # logger.info(f" entropy_encoded_prediction_data  len : {len(self.entropy_encoded_prediction_data)}, {len(self.entropy_encoded_prediction_data) // 8}")

    def entropy_decode_prediction_data(self, enc, params: InputParameters):
        width_ = params.width
        height_ = params.height
        block_size_ = params.encoder_config.block_size

        self.mv_field = {}  # Initialize an empty dictionary to store the decoded motion vectors

        bitstream = bitarray()
        bitstream.frombytes(enc)

        # Initialize an index to track the current block
        index = 0

        # Decode each pair of motion vector components (mv_x and mv_y) from the bitstream
        while bitstream:
            try:
                # Decode the first component of the motion vector (mv_x)
                mv_x, bitstream = exp_golomb_decode(bitstream)
                if not bitstream:
                    logger.debug(f"bitstream empty, breaking.")
                    break
                mv_y, bitstream = exp_golomb_decode(bitstream)
                if params.encoder_config.nRefFrames > 1 :
                    mv_ref_frame_idx, bitstream = exp_golomb_decode(bitstream)
                else:
                    mv_ref_frame_idx = 0

                # Calculate the pixel coordinates (column_index, row_index) for the block's top-left corner
                row_index = (index // (width_ // block_size_)) * block_size_  # Y-coordinate
                column_index = (index % (width_ // block_size_)) * block_size_  # X-coordinate

                # Ensure the calculated coordinates are within the frame dimensions
                if row_index < height_ and column_index < width_:
                    # Store the motion vector in the dictionary with the (column_index, row_index) as the key
                    self.mv_field[(column_index, row_index)] = [mv_x, mv_y, mv_ref_frame_idx]
                else:
                    logger.warn(f"Warning: Calculated coordinates {(column_index, row_index)} are out of bounds.")

                index += 1  # Move to the next block

            except ValueError:
                # If there's an issue in decoding (e.g., insufficient bits), exit the loop
                print("Decoding error or incomplete bitstream.")
                break

        return self.mv_field


def construct_frame_from_dct_and_mv(quant_dct_coff_frame, reference_frames, mv_field, encoder_config: EncoderConfig):
    block_size = encoder_config.block_size
    quantization_factor = encoder_config.quantization_factor
    height, width = reference_frames[0].shape
    decoded_frame = np.zeros_like(reference_frames[0], dtype=np.uint8)

    # Generate the quantization matrix Q based on block size and quantization factor
    Q = generate_quantization_matrix(block_size, quantization_factor)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            check_index_out_of_bounds(x, y, mv_field.get((x, y)), width, height, block_size)
            # Get the quantized residual block
            dct_coffs_block = quant_dct_coff_frame[y:y + block_size, x:x + block_size]

            # Rescale the residual block by multiplying by Q
            rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)

            # Apply inverse DCT to the rescaled residual block
            idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)

            # Get the predicted block using the motion vector
            predicted_b = find_mv_predicted_block(mv_field.get((x, y), None), x, y, reference_frames, block_size)

            # Check if the predicted block is valid
            mv = mv_field.get((x, y))
            if predicted_b is None or predicted_b.size == 0:
                pred_x = x + mv[0]
                pred_y = y + mv[1]

                logger.warning(f"predicted_b is None"
                            f"\n\tpredicted_b @ [{y:3}, {x:3}] -> {predicted_b}\n"
                            f"\tmv -> {mv}\n"
                            f"\tprev_frame[{pred_y}:{pred_y + block_size}, {pred_x}:{pred_x + block_size}]")

                # Handle the case where predicted_b is invalid or empty
                predicted_b = np.zeros((block_size, block_size), dtype=np.int16)

            # Ensure the predicted block size matches the residual block size
            if predicted_b.shape != idct_residual_block.shape:
                logger.warning(f"predicted_b  @ [{y:3}, {x:3}] {predicted_b.shape} != idct_residual_block {idct_residual_block.shape}\n"
                            f"\tmv -> {mv}\n")
                # Adjust the shape of the predicted block to match the residual block
                predicted_b = np.pad(predicted_b,
                                     ((0, idct_residual_block.shape[0] - predicted_b.shape[0]),
                                      (0, idct_residual_block.shape[1] - predicted_b.shape[1])),
                                     mode='edge')  # Padding to match the size

            # Reconstruct the block by adding the predicted block and the rescaled residual
            decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)

            # Clip values to avoid overflow/underflow and convert back to uint8
            decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)

            # Place the reconstructed block in the decoded frame
            decoded_frame[y:y + block_size, x:x + block_size] = decoded_block

    return decoded_frame


def check_index_out_of_bounds(x, y, motion_vector, width, height, block_size):

    if x + motion_vector[0] < 0 or y + motion_vector[1] < 0 :
        logger.error(f" mv [{motion_vector}] for [{x}, {y}] referencing small value [{x + motion_vector[0]}] or [{y + motion_vector[1] < 0}]")
    if x + motion_vector[0] + block_size > width or y + motion_vector[1] + block_size > height:
        logger.error(f" mv [{motion_vector}] for [{x}, {y}] referencing large value [{x + motion_vector[0] + block_size}]  or [{y + motion_vector[1] + block_size}]")
