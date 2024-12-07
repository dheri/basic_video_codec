import numpy as np
from bitarray import bitarray

from common import get_logger
from encoder.Frame import Frame, apply_dct_and_quantization, reconstruct_block
from encoder.PredictionMode import PredictionMode
from encoder.RateControl.RateControl import find_rc_qp_for_row, calculate_constant_row_bit_budget, \
    calculate_proportional_row_bit_budget
from encoder.dct import generate_quantization_matrix, rescale_block, apply_idct_2d
from encoder.entropy_encoder import exp_golomb_encode, exp_golomb_decode
from encoder.params import EncoderConfig, EncodedIBlock

logger = get_logger()


class IFrame(Frame):
    def __init__(self, curr_frame=None):
        super().__init__(curr_frame)
        self.prediction_mode = PredictionMode.INTRA_FRAME
        self.intra_modes = None

    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        curr_frame = self.curr_frame
        block_size = encoder_config.block_size
        height, width = curr_frame.shape

        mae_of_blocks = 0
        self.intra_modes = []  # To store the intra prediction modes (0 for horizontal, 1 for vertical)
        self.reconstructed_frame = np.zeros_like(curr_frame)
        residual_w_mc_frame = np.zeros_like(curr_frame)
        self.quantized_dct_residual_frame = np.zeros_like(curr_frame, dtype=np.int16)

        prev_rc_qp = encoder_config.quantization_factor
        rc_qp = encoder_config.quantization_factor
        # Loop through each row in the frame
        for y in range(0, height, block_size):
            row_idx = y//block_size
            if encoder_config.RCflag:
                row_bit_budget = 0

                if encoder_config.RCflag == 1:
                    row_bit_budget = calculate_constant_row_bit_budget(self.bit_budget, row_idx, encoder_config)
                    rc_qp = find_rc_qp_for_row(row_bit_budget, encoder_config.rc_lookup_table, 'I')
                if encoder_config.RCflag == 2 and not self.is_first_pass:
                    # TODO: This is second pass
                    # row_bit_budget = calculate_proportional_row_bit_budget(self, row_idx, encoder_config)
                    row_bit_budget = calculate_constant_row_bit_budget(self.bit_budget, row_idx, encoder_config)

                    # rc_qp = find_rc_qp_for_row(row_bit_budget, encoder_config.rc_lookup_table, 'I')
                    # logger.info(f"rc_qp == {rc_qp} for {row_bit_budget:7.2f} / [{self.bit_budget:9.2f}]")

                # logger.info(f"[{row_idx:2d}] f_bb [{self.bit_budget:9.2f}] row_bb [{row_bit_budget:8.2f}] , qp=[{rc_qp}]")
            #  Loop through each block  in the row
            for x in range(0, width, block_size):
                curr_block = curr_frame[y:y + block_size, x:x + block_size]

                encoded_block = process_block(
                    curr_block, self.reconstructed_frame, x, y, block_size, rc_qp
                )

                # Store intra mode and update MAE
                self.intra_modes.append(encoded_block.mode)
                mae_of_blocks += encoded_block.mae

                # Update reconstructed frame and quantized residuals
                self.reconstructed_frame[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block
                self.quantized_dct_residual_frame[y:y + block_size,
                x:x + block_size] = encoded_block.quantized_dct_coffs  # quantized_dct_residual_block
                residual_w_mc_frame[y:y + block_size,
                x:x + block_size] = encoded_block.residual_block_wo_mc  # residual_block
                self.total_mae_comparisons += encoded_block.mae_comparisons_to_encode
            rc_qp_diff = rc_qp - prev_rc_qp
            self.entropy_encode_prediction_data_row(row_idx, encoder_config, rc_qp_diff)
            self.entropy_encode_dct_coffs_row(row_idx, encoder_config)
            row_bits_consumed = (len(self.entropy_encoded_DCT_coffs) - self.entropy_encoded_dct_length
                             + len(self.entropy_encoded_prediction_data) - self.entropy_encoded_prediction_data_length
                             # + 8 * 6
                                 )
            self.bit_budget -= row_bits_consumed
            self.bits_per_row.append(row_bits_consumed)
            self.entropy_encoded_dct_length = len(self.entropy_encoded_DCT_coffs)
            self.entropy_encoded_prediction_data_length = len(self.entropy_encoded_prediction_data)
            # TODO: Verify if prev_rc_qp needs to be uncommented
            # prev_rc_qp = rc_qp

        avg_mae = mae_of_blocks / ((height // block_size) * (width // block_size))
        # self.reconstructed_frame = reconstructed_frame
        # self.quantized_dct_residual_frame = quantized_dct_residual_frame
        # self.intra_modes = intra_modes
        self.avg_mae = avg_mae
        self.residual_frame = residual_w_mc_frame
        # doesnt make sense for w/o mc in INTRA
        self.residual_wo_mc_frame = residual_w_mc_frame

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        block_size = encoder_config.block_size
        height, width = frame_shape
        reconstructed_frame = np.zeros((height, width), dtype=np.uint8)
        Q_map = {}

        # Iterate over blocks to reconstruct the frame
        for y in range(0, height, block_size):
            row_idx = y//block_size
            rc_qp = self.rc_qp_per_row[row_idx]
            if not rc_qp in Q_map:
                Q_map[rc_qp] = generate_quantization_matrix(block_size, rc_qp)
            Q = Q_map[rc_qp]

            for x in range(0, width, block_size):
                dct_coffs_block = self.quantized_dct_residual_frame[y:y + block_size, x:x + block_size]
                # logger.info(f" dct_coffs_block extremes : [{np.min(dct_coffs_block)}, {np.max(dct_coffs_block)} ]")

                rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)
                idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)
                predicted_b = find_intra_predict_block(self.intra_modes[(y // encoder_config.block_size) * (
                        width // encoder_config.block_size) + (x // encoder_config.block_size)],
                                                       reconstructed_frame, x, y, encoder_config.block_size)

                decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)
                decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)
                reconstructed_frame[y:y + encoder_config.block_size, x:x + encoder_config.block_size] = decoded_block

        self.curr_frame = reconstructed_frame
        return reconstructed_frame  # This should be the reconstructed frame

    def entropy_encode_prediction_data_row(self, row_idx, ec:EncoderConfig, rc_qp_diff):
        if not self.entropy_encoded_prediction_data:
            self.entropy_encoded_prediction_data = bitarray()

        enc = exp_golomb_encode(rc_qp_diff)
        self.entropy_encoded_prediction_data.extend(enc)

        blocks_per_row = ec.resolution[0] // ec.block_size

        start_idx = row_idx * blocks_per_row
        end_idx = start_idx + blocks_per_row

        for m in (self.intra_modes[start_idx:end_idx]):
            enc = exp_golomb_encode(m)
            self.entropy_encoded_prediction_data.extend(enc)

    # def entropy_encode_prediction_data(self, encoder_config:EncoderConfig):
    #     self.entropy_encoded_prediction_data = bitarray()
    #     # logger.info(self.intra_modes)
    #     for m in self.intra_modes:
    #         enc = exp_golomb_encode(m)
    #         self.entropy_encoded_prediction_data.extend(enc)
    #     # logger.info(f" entropy_encoded_prediction_data  len : {len(self.entropy_encoded_prediction_data)}, {len(self.entropy_encoded_prediction_data) // 8}")
    #     # logger.info(self.entropy_encoded_prediction_data)

    def entropy_decode_prediction_data(self, enc, params=None):
        decoded_modes = []
        bitstream = bitarray()  # Ensure `enc` is a bitarray
        bitstream.frombytes(enc)
        ec = params.encoder_config
        prev_rq_qp = ec.quantization_factor
        blocks_in_row = ec.resolution[0] // ec.block_size
        num_of_rows = ec.resolution[1] // ec.block_size



        blocks_processed = 0
        # Decode each encoded mode
        while bitstream:
            try:
                if blocks_processed % blocks_in_row == 0:
                    if len(self.rc_qp_per_row) >= num_of_rows:
                        break
                    rc_qp_diff, bitstream = exp_golomb_decode(bitstream)
                    rc_qp = prev_rq_qp + rc_qp_diff
                    self.rc_qp_per_row.append(rc_qp)

                decoded_value, bitstream = exp_golomb_decode(bitstream)
                decoded_modes.append(decoded_value)
                blocks_processed += 1
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
