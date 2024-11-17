from collections import OrderedDict

import numpy as np
from bitarray import bitarray

from common import get_logger
from encoder.Frame import Frame, apply_dct_and_quantization, reconstruct_block
from encoder.PredictionMode import PredictionMode
from encoder.block_predictor import find_lowest_mae_block, find_fast_me_block, build_pre_interpolated_buffer, \
    get_ref_block_at_mv
from encoder.dct import generate_quantization_matrix, rescale_block, apply_idct_2d
from encoder.entropy_encoder import exp_golomb_encode, exp_golomb_decode
from encoder.params import EncoderConfig, EncodedPBlock
from input_parameters import InputParameters

logger = get_logger()


class PFrame(Frame):
    def __init__(self, curr_frame=None, reference_frames=None, interpolated_reference_frames=None):
        super().__init__(curr_frame, reference_frames, interpolated_reference_frames)
        self.prediction_mode = PredictionMode.INTER_FRAME
        self.mv_field = None
        self.avg_mae = None

    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        block_size = encoder_config.block_size

        height, width = self.curr_frame.shape
        num_of_blocks = (height // block_size) * (width // block_size)
        mv_field = {(0,0) : [0,0]}
        mae_of_blocks = 0

        # Initialize output frames
        reconstructed_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.uint8)
        residual_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        residual_frame_wo_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        quat_dct_coffs_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int16)
        prev_processed_block_cords = (0,0)
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                encoded_block =self.process_block(x, y, width, height, mv_field, prev_processed_block_cords, encoder_config)
                block_cords = encoded_block.block_coords
                x, y = block_cords

                # Update frames with the encoded block data
                reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
                residual_frame_wo_mc[y:y + block_size, x:x + block_size] = encoded_block.residual_block_wo_mc
                quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

                mae_of_blocks += encoded_block.mae
                self.total_mae_comparisons += encoded_block.mae_comparisons_to_encode
                prev_processed_block_cords = block_cords

        avg_mae = mae_of_blocks / num_of_blocks

        sorted_mv_field = OrderedDict(sorted(mv_field.items(), key=lambda item: (item[0][1], item[0][0])))
        self.mv_field = sorted_mv_field  # Populate the motion vector field

        self.avg_mae = avg_mae
        self.residual_frame = residual_frame_with_mc
        self.residual_wo_mc_frame = residual_frame_wo_mc
        self.quantized_dct_residual_frame = quat_dct_coffs_frame_with_mc
        self.reconstructed_frame = reconstructed_frame_with_mc
        return self

    def process_block(self, x, y, width, height, mv_field, prev_processed_block_cords, encoder_config) -> EncodedPBlock:
        block_size, search_range, quantization_factor = encoder_config.block_size, encoder_config.search_range, encoder_config.quantization_factor
        curr_block = self.curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)
        # consider latest ref frame for no mv
        prev_block = self.reference_frames[0][y:y + block_size, x:x + block_size].astype(np.int16)

        mvp = mv_field[prev_processed_block_cords]
        motion_vector, best_match_mae, _ , comparisons = self.get_motion_vector(curr_block, ( x, y), mvp , encoder_config)
        mv_field[(x, y)] = motion_vector

        # Generate residual and predicted block
        predicted_block_with_mc, residual_block_with_mc = self.generate_residual_block(curr_block, ( x, y),
                                                                                  motion_vector, encoder_config)
        residual_block_wo_mc = np.subtract(curr_block, prev_block)
        # Apply DCT and quantization
        quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

        # Reconstruct the block using the predicted and inverse DCT
        clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q,
                                                                             predicted_block_with_mc)
        check_index_out_of_bounds(x, y, motion_vector,width,height, block_size)

        return EncodedPBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block,
                             residual_block_wo_mc, clipped_reconstructed_block, comparisons)

    def get_motion_vector(self, curr_block, curr_block_cords, mvp, ec:EncoderConfig):
        if ec.fastME:
            return find_fast_me_block(curr_block, curr_block_cords, mvp, self, ec, 0)
        else:
            return find_lowest_mae_block(curr_block, curr_block_cords, self, ec)

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        return construct_frame_from_dct_and_mv(self, encoder_config)

    def entropy_encode_prediction_data(self, encoder_config:EncoderConfig):
        self.entropy_encoded_prediction_data = bitarray()
        prev_mv = (0,0,0)
        for key, mv in self.mv_field.items():
            enc_mv_x = exp_golomb_encode(mv[0] - prev_mv[0] )
            self.entropy_encoded_prediction_data.extend(enc_mv_x)
            enc_mv_y = exp_golomb_encode(mv[1] - prev_mv[1] )
            self.entropy_encoded_prediction_data.extend(enc_mv_y)
            if encoder_config.nRefFrames>1:
                enc_mv_ref_frame_idx = exp_golomb_encode(mv[2] - prev_mv[2])
                self.entropy_encoded_prediction_data.extend(enc_mv_ref_frame_idx)
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()} {enc_mv_ref_frame_idx.to01()}]")
            else:
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()}]")
            prev_mv = mv

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
        prev_mv = (0,0,0)

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

                mv = (prev_mv[0] + mv_x, prev_mv[1] + mv_y, prev_mv[2] + mv_ref_frame_idx)
                # Calculate the pixel coordinates (column_index, row_index) for the block's top-left corner
                row_index = (index // (width_ // block_size_)) * block_size_  # Y-coordinate
                column_index = (index % (width_ // block_size_)) * block_size_  # X-coordinate

                # Ensure the calculated coordinates are within the frame dimensions
                if row_index < height_ and column_index < width_:
                    # Store the motion vector in the dictionary with the (column_index, row_index) as the key
                    self.mv_field[(column_index, row_index)] = mv
                else:
                    logger.warn(f"Warning: Calculated coordinates {(column_index, row_index)} are out of bounds.")

                index += 1  # Move to the next block
                prev_mv = mv

            except ValueError:
                # If there's an issue in decoding (e.g., insufficient bits), exit the loop
                print("Decoding error or incomplete bitstream.")
                break

        return self.mv_field

    def find_mv_predicted_block(self, mv, curr_block_cords, ec:EncoderConfig):
        x, y = curr_block_cords
        reference_frames = self.reference_frames

        mv_x = x + mv[0]
        mv_y = y + mv[1]

        if len(reference_frames) > 1:
            ref_frame_idx = mv[2]
        else:
            ref_frame_idx = 0

        predicted_block = get_ref_block_at_mv(
            self.reference_frames[ref_frame_idx],
            self.interpolated_reference_frames[ref_frame_idx],
            curr_block_cords, mv_x, mv_y, ec)

        assert predicted_block.shape == (ec.block_size, ec.block_size)

        return predicted_block

    def generate_residual_block(self, curr_block, curr_block_cords, mv, ec:EncoderConfig):
        predicted_block_with_mc = self.find_mv_predicted_block(mv, curr_block_cords, ec).astype(np.int16)
        residual_block_with_mc = np.subtract(curr_block.astype(np.int16), predicted_block_with_mc.astype(np.int16))
        return predicted_block_with_mc, residual_block_with_mc


def construct_frame_from_dct_and_mv( frame:PFrame,  encoder_config: EncoderConfig):
    reference_frames = frame.reference_frames
    quant_dct_coff_frame = frame.quantized_dct_residual_frame
    mv_field = frame.mv_field

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
            predicted_b = frame.find_mv_predicted_block(mv_field.get((x, y)), (x, y), encoder_config)

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


