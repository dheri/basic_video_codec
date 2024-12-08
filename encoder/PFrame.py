from collections import OrderedDict
from statistics import mean

import numpy as np
from bitarray import bitarray

from common import get_logger
from encoder.Frame import Frame, apply_dct_and_quantization, reconstruct_block
from encoder.PredictionMode import PredictionMode
from encoder.RateControl.RateControl import calculate_constant_row_bit_budget, find_rc_qp_for_row, \
    calculate_proportional_row_bit_budget
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
        self.mv_field :dict  = {}
        self.avg_mae = None

    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        block_size = encoder_config.block_size

        height, width = self.curr_frame.shape
        num_of_blocks = (height // block_size) * (width // block_size)
        mv_field = {(0,0) : [0,0]}
        mae_of_blocks = 0

        # Initialize output frames
        self.reconstructed_frame = np.zeros_like(self.curr_frame, dtype=np.uint8)
        residual_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        residual_frame_wo_mc = np.zeros_like(self.curr_frame, dtype=np.int8)
        self.quantized_dct_residual_frame = np.zeros_like(self.curr_frame, dtype=np.int16)
        prev_rc_qp = encoder_config.quantization_factor
        rc_qp = encoder_config.quantization_factor
        prev_processed_block_cords = (0,0)
        prev_frame_avg_qp = int(mean(self.prev_frame.rc_qp_per_row) - 0.1) + 1 # a ceil fn with offset of 0.1
        # logger.info(f"{self.index:2d}: prev_f_avg_qp {'f' if self.is_first_pass else 's'} = {mean(self.prev_frame.rc_qp_per_row):4.2f} | {prev_frame_avg_qp} : {self.prev_frame.rc_qp_per_row}")
        prev_pass_mv_field = []
        if encoder_config.RCflag == 3 and not self.is_first_pass:
            prev_pass_mv_field = self.prev_pass_frame.mv_field

        for y in range(0, height, block_size):
            row_idx = y//block_size
            rc_qp = self.get_rc_qp(encoder_config, prev_frame_avg_qp, rc_qp, row_idx)
            for x in range(0, width, block_size):
                encoded_block =self.process_block(x, y, width, height, mv_field, prev_pass_mv_field, prev_processed_block_cords, encoder_config, rc_qp)
                block_cords = encoded_block.block_coords
                x, y = block_cords

                # Update frames with the encoded block data
                self.reconstructed_frame[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
                residual_frame_wo_mc[y:y + block_size, x:x + block_size] = encoded_block.residual_block_wo_mc
                self.quantized_dct_residual_frame[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

                mae_of_blocks += encoded_block.mae
                self.total_mae_comparisons += encoded_block.mae_comparisons_to_encode
                prev_processed_block_cords = block_cords
            sorted_mv_field = OrderedDict(sorted(mv_field.items(), key=lambda item: (item[0][1], item[0][0])))
            self.mv_field.update(sorted_mv_field)
            rc_qp_diff = rc_qp - prev_rc_qp

            self.entropy_encode_prediction_data_row(row_idx, encoder_config, rc_qp_diff)
            self.entropy_encode_dct_coffs_row(row_idx, encoder_config)
            row_bits_consumed = (len(self.entropy_encoded_DCT_coffs) - self.entropy_encoded_dct_length
                             + len(self.entropy_encoded_prediction_data) - self.entropy_encoded_prediction_data_length
                             # + (8 * 6)
                                 )
            self.bit_budget -= row_bits_consumed
            self.bits_per_row.append(row_bits_consumed)
            self.entropy_encoded_dct_length = len(self.entropy_encoded_DCT_coffs)
            self.entropy_encoded_prediction_data_length = len(self.entropy_encoded_prediction_data)

        logger.info(f"{self.index:2d}: prev_f_avg_qp {'f' if self.is_first_pass else 's'} = {mean(self.prev_frame.rc_qp_per_row):4.2f} | {prev_frame_avg_qp} : {self.rc_qp_per_row}")

        avg_mae = mae_of_blocks / num_of_blocks
        # sorted_mv_field = OrderedDict(sorted(mv_field.items(), key=lambda item: (item[0][1], item[0][0])))
        # self.mv_field = sorted_mv_field  # Populate the motion vector field

        self.avg_mae = avg_mae
        self.residual_frame = residual_frame_with_mc
        self.residual_wo_mc_frame = residual_frame_wo_mc
        # self.quantized_dct_residual_frame = quat_dct_coffs_frame_with_mc
        # self.reconstructed_frame = reconstructed_frame_with_mc
        return self

    def process_block(self, x, y, width, height, mv_field, prev_pass_mv_field, prev_processed_block_cords, encoder_config, rc_qp) -> EncodedPBlock:
        block_size, search_range, quantization_factor = encoder_config.block_size, encoder_config.search_range, rc_qp
        curr_block = self.curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)
        # consider latest ref frame for no mv
        prev_block = self.reference_frames[0][y:y + block_size, x:x + block_size].astype(np.int16)

        mvp = mv_field[prev_processed_block_cords]
        # if encoder_config.RCflag == 3: # get mvp of block from last pass
        #     mvp = prev_pass_mv_field[prev_processed_block_cords]
        mv, best_match_mae, _ , comparisons = self.get_motion_vector(curr_block, ( x, y), mvp , encoder_config)
        # logger.debug(f"b ({x:3} {y:3}) mae [{best_match_mae:>6.2f}] mv:{mv} ")
        mv_field[(x, y)] = mv


        # Generate residual and predicted block
        predicted_block_with_mc, residual_block_with_mc = self.generate_residual_block(curr_block, ( x, y),
                                                                                  mv, encoder_config)
        residual_block_wo_mc = np.subtract(curr_block, prev_block)
        # Apply DCT and quantization
        quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

        # Reconstruct the block using the predicted and inverse DCT
        clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q,
                                                                             predicted_block_with_mc)

        return EncodedPBlock((x, y), mv, best_match_mae, quantized_dct_coffs, idct_residual_block,
                             residual_block_wo_mc, clipped_reconstructed_block, comparisons)

    def get_motion_vector(self, curr_block, curr_block_cords, mvp, ec:EncoderConfig):
        if ec.fastME:
            return find_fast_me_block(curr_block, curr_block_cords, mvp, self, ec, 0)
        else:
            return find_lowest_mae_block(curr_block, curr_block_cords, self, ec)

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        return construct_frame_from_dct_and_mv(self, encoder_config)

    def entropy_encode_prediction_data_row(self, row_idx, ec:EncoderConfig, rc_qp_diff):
        blocks_per_row = ec.resolution[0] // ec.block_size
        block_y = row_idx * ec.block_size

        prev_mv = (0,0,0)
        if not self.entropy_encoded_prediction_data:
            self.entropy_encoded_prediction_data = bitarray()
        else:
            prev_mv = self.mv_field[(ec.resolution[0] - ec.block_size, block_y - ec.block_size)]

        enc = exp_golomb_encode(rc_qp_diff)
        self.entropy_encoded_prediction_data.extend(enc)


        for block_x in range(0, ec.resolution[0], ec.block_size):
            key = (block_x, block_y)
            mv = self.mv_field[key]
            enc_mv_x = exp_golomb_encode(mv[0] - prev_mv[0] )
            self.entropy_encoded_prediction_data.extend(enc_mv_x)
            enc_mv_y = exp_golomb_encode(mv[1] - prev_mv[1] )
            self.entropy_encoded_prediction_data.extend(enc_mv_y)
            if ec.nRefFrames>1:
                enc_mv_ref_frame_idx = exp_golomb_encode(mv[2] - prev_mv[2])
                self.entropy_encoded_prediction_data.extend(enc_mv_ref_frame_idx)
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()} {enc_mv_ref_frame_idx.to01()}]")
            else:
                logger.debug(f" {key} : {mv} -> [{enc_mv_x.to01()} {enc_mv_y.to01()}]")
            prev_mv = mv


    def entropy_decode_prediction_data(self, enc, params: InputParameters):
        width = params.width
        height = params.height
        block_size_ = params.encoder_config.block_size

        self.mv_field = {}  # Initialize an empty dictionary to store the decoded motion vectors

        bitstream = bitarray()
        bitstream.frombytes(enc)

        # Initialize blocks_processed to track the current block
        blocks_processed = 0
        prev_mv = (0,0,0)

        ec = params.encoder_config
        prev_rq_qp = ec.quantization_factor
        blocks_in_row = ec.resolution[0] // ec.block_size
        num_of_rows = ec.resolution[1] // ec.block_size
        # blocks_processed = 0

        # Decode each pair of motion vector components (mv_x and mv_y) from the bitstream
        while bitstream:
            try:
                if blocks_processed % blocks_in_row == 0: # start of each row
                    if len(self.rc_qp_per_row) >= num_of_rows:  # start of last row
                        logger.debug(f"found last row {len(self.rc_qp_per_row)} >= {num_of_rows}")
                        break
                    rc_qp_diff, bitstream = exp_golomb_decode(bitstream)
                    rc_qp = prev_rq_qp + rc_qp_diff
                    self.rc_qp_per_row.append(rc_qp)

                # Decode the first component of the motion vector (mv_x)
                mv_x, bitstream = exp_golomb_decode(bitstream)
                if not bitstream:
                    logger.info(f"bitstream empty, breaking.")
                    break
                mv_y, bitstream = exp_golomb_decode(bitstream)
                if params.encoder_config.nRefFrames > 1 :
                    mv_ref_frame_idx, bitstream = exp_golomb_decode(bitstream)
                else:
                    mv_ref_frame_idx = 0

                mv = (prev_mv[0] + mv_x, prev_mv[1] + mv_y, prev_mv[2] + mv_ref_frame_idx)
                # Calculate the pixel coordinates (column_index, row_index) for the block's top-left corner
                row_index = (blocks_processed // (width // block_size_)) * block_size_  # Y-coordinate
                column_index = (blocks_processed % (width // block_size_)) * block_size_  # X-coordinate

                # Ensure the calculated coordinates are within the frame dimensions
                if row_index < height and column_index < width:
                    # Store the motion vector in the dictionary with the (column_index, row_index) as the key
                    self.mv_field[(column_index, row_index)] = mv
                else:
                    logger.warn(f"Warning: Calculated coordinates {(column_index, row_index)} are out of bounds.")

                blocks_processed += 1  # Move to the next block
                prev_mv = mv

            except ValueError:
                # If there's an issue in decoding (e.g., insufficient bits), exit the loop
                print("Decoding error or incomplete bitstream.")
                break

        return self.mv_field

    def find_mv_predicted_block(self, mv, curr_block_cords, ec:EncoderConfig):
        reference_frames = self.reference_frames
        if len(reference_frames) > 1:
            ref_frame_idx = mv[2]
        else:
            ref_frame_idx = 0

        predicted_block = get_ref_block_at_mv(
            self.reference_frames[ref_frame_idx],
            self.interpolated_reference_frames[ref_frame_idx],
            curr_block_cords, mv[0], mv[1], ec)

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

    Q_map = {}

    for y in range(0, height, block_size):
        row_idx = y // block_size
        rc_qp = frame.rc_qp_per_row[row_idx]
        if not rc_qp in Q_map:
            Q_map[rc_qp] = generate_quantization_matrix(block_size, rc_qp)
        Q = Q_map[rc_qp]

        for x in range(0, width, block_size):
            check_index_out_of_bounds(x, y, mv_field.get((x, y)), width, height, encoder_config)
            # Get the quantized residual block
            dct_coffs_block = quant_dct_coff_frame[y:y + block_size, x:x + block_size]

            # Rescale the residual block by multiplying by Q
            rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)

            # Apply inverse DCT to the rescaled residual block
            idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)

            # Get the predicted block using the motion vector
            mv = mv_field.get((x, y))
            predicted_b = frame.find_mv_predicted_block(mv, (x, y), encoder_config)

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


def check_index_out_of_bounds(x, y, motion_vector, width, height, ec:EncoderConfig):
    block_size = ec.block_size
    if ec.fastME:
        motion_vector = (motion_vector[0]/2, motion_vector[1]/2, motion_vector[2]/2)

    if x + motion_vector[0] < 0 or y + motion_vector[1] < 0 :
        logger.error(f" mv [{motion_vector}] for [{x}, {y}] referencing small value [{x + motion_vector[0]}] or [{y + motion_vector[1] }]")
        return True
    if x + motion_vector[0] + block_size > width or y + motion_vector[1] + block_size > height:
        logger.error(f" mv [{motion_vector}] for [{x}, {y}] referencing large value [{x + motion_vector[0] + block_size}]  or [{y + motion_vector[1] + block_size}]")
        return True
    return False


