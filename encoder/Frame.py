from collections import deque
from typing import Optional

import numpy as np
from bitarray import bitarray
from skimage.metrics import peak_signal_noise_ratio

from common import get_logger, split_into_blocks, merge_blocks, pad_with_zeros
from encoder.PredictionMode import PredictionMode
from encoder.dct import apply_dct_2d, generate_quantization_matrix, quantize_block, rescale_block, apply_idct_2d
from encoder.entropy_encoder import zigzag_order, rle_encode, exp_golomb_encode, exp_golomb_decode, rle_decode, \
    inverse_zigzag_order
from encoder.params import EncoderConfig
from file_io import write_y_only_frame, write_mv_to_file
from input_parameters import InputParameters

logger = get_logger()


class Frame:
    EOB_MARKER = 8190

    def __init__(self, curr_frame=None, reference_frames=None, interpolated_reference_frames=None ):
        self.reference_frames : deque = reference_frames
        self.interpolated_reference_frames : deque = interpolated_reference_frames
        self.curr_frame = curr_frame
        self.prediction_mode: PredictionMode = PredictionMode.INTER_FRAME
        self.entropy_encoded_prediction_data: Optional[bitarray] = None
        self.entropy_encoded_DCT_coffs: Optional[bitarray] = None

        self.residual_frame = None
        self.residual_wo_mc_frame = None
        self.quantized_dct_residual_frame = None
        self.reconstructed_frame = None
        self.avg_mae = None
        self.total_mae_comparisons = 0
        self.bit_budget = 0
        self.entropy_encoded_dct_length=0
        self.entropy_encoded_prediction_data_length=0
        self.rc_qp_per_row = []
        self.bits_per_row = []
        self.is_first_pass = True
        self.prev_pass_frame : Frame | None = None
        self.index = 0
        self.scaling_factor = 1
    def encode_mc_q_dct(self, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def decode_mc_q_dct(self, frame_shape, encoder_config: EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def entropy_encode_prediction_data(self, encoder_config:EncoderConfig):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def entropy_decode_prediction_data(self, enc, params: InputParameters):
        raise NotImplementedError(f"{type(self)} need to be overridden")

    def entropy_encode_dct_coffs_row(self, row_idx, ec:EncoderConfig):
        if not self.entropy_encoded_DCT_coffs:
            self.entropy_encoded_DCT_coffs = bitarray()
        start_row = row_idx * ec.block_size
        end_row = start_row + ec.block_size

        dct_blocks_row = self.quantized_dct_residual_frame[start_row: end_row, :]
        blocks = split_into_blocks(dct_blocks_row, ec.block_size)
        for block in blocks:
            zigzag_dct_coffs = zigzag_order(block)
            rle = rle_encode(zigzag_dct_coffs)
            for symbol in rle:
                enc = exp_golomb_encode(symbol)
                self.entropy_encoded_DCT_coffs.extend(enc)
            self.entropy_encoded_DCT_coffs.extend(exp_golomb_encode(Frame.EOB_MARKER))

        # logger.info(f"prediction : {len(self.entropy_encoded_prediction_data):6d} |  DCT: {len(self.entropy_encoded_DCT_coffs):6d}")



    def entropy_decode_dct_coffs(self, params: InputParameters):
        block_size = params.encoder_config.block_size
        rle_blocks = []
        decoded_blocks = []
        # bitstream = self.entropy_encoded_DCT_coffs
        bit_array = bitarray()
        bit_array.frombytes(self.entropy_encoded_DCT_coffs)

        rle_decoded = []

        # Step 1: Decode the Exponential-Golomb encoded symbols to construct rle blocks
        while bit_array:
            symbol, bit_array = exp_golomb_decode(bit_array)
            if symbol == Frame.EOB_MARKER:
                rle_blocks.append(rle_decoded)
                rle_decoded = []  # Reset for the next block
                continue
            rle_decoded.append(symbol)

        # Step 2: Apply RLE decoding to reconstruct the zigzag order of coefficients
        for rle_block in rle_blocks:
            decoded_coffs = rle_decode(rle_block)
            pad_with_zeros(decoded_coffs, block_size ** 2)
            block = inverse_zigzag_order(decoded_coffs, block_size)
            decoded_blocks.append(block)

        # Step 4: Reconstruct the frame from the blocks
        self.quantized_dct_residual_frame = merge_blocks(decoded_blocks, block_size, (params.height, params.width))

        return self.quantized_dct_residual_frame

    def write_metrics_data(self, metrics_csv_writer, frame_index, encoder_config: EncoderConfig):
        psnr = peak_signal_noise_ratio(self.curr_frame, self.reconstructed_frame)
        dct_coffs_extremes = self.get_quat_dct_coffs_extremes()
        logger.info(
            f" {self.prediction_mode:1} {frame_index:2}: i={encoder_config.block_size} r={encoder_config.search_range}, qp={encoder_config.quantization_factor}, mae[{round(self.avg_mae, 2):7.2f}] psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")
        metrics_csv_writer.writerow([frame_index, self.avg_mae, psnr])

    def write_encoded_to_file(self, mv_fh, quant_dct_coff_fh, residual_yuv_fh, residual_wo_mc_yuv_fh, reconstructed_fh,
                              encoder_config):

        write_y_only_frame(residual_yuv_fh, self.residual_frame)
        write_y_only_frame(residual_wo_mc_yuv_fh, self.residual_wo_mc_frame)
        write_y_only_frame(quant_dct_coff_fh, self.quantized_dct_residual_frame)
        write_y_only_frame(reconstructed_fh, self.reconstructed_frame)

        if self.prediction_mode == PredictionMode.INTER_FRAME:
            write_mv_to_file(mv_fh, self.mv_field)
        else:
            mv_fh.write('\n')

    def get_quat_dct_coffs_extremes(self):
        # Ensure quat_dct_coffs_with_mc is a numpy array to use numpy's min/max
        if isinstance(self.quantized_dct_residual_frame, np.ndarray):
            min_value = np.min(self.quantized_dct_residual_frame)
            max_value = np.max(self.quantized_dct_residual_frame)
            return [min_value, max_value]
        else:
            raise TypeError(f"{self.quantized_dct_residual_frame} quantized_dct_residual_frame must be a numpy array")

    def get_mv_extremes(self):
        if self.prediction_mode == PredictionMode.INTER_FRAME:
            # Convert mv_field.values() to a NumPy array for easier column-wise min/max computation
            mv_array = np.array(list(self.mv_field.values()))  # Shape: (num_vectors, 3)

            # Compute min and max for each component
            min_values = mv_array.min(axis=0)  # Minimum along each column
            max_values = mv_array.max(axis=0)  # Maximum along each column

            return [min_values.tolist(), max_values.tolist()]  # Convert NumPy arrays to lists for consistency
        else:
            # Min/max for intra_modes
            return [np.min(self.intra_modes), np.max(self.intra_modes)]

    def get_overage_ratios(self, encoder_config:EncoderConfig):
        if not self.is_first_pass:
            raise ValueError("why is overage being called in first pass?")
        frame_bits_consumed = (len(self.entropy_encoded_DCT_coffs) + len(self.entropy_encoded_prediction_data)+ 8 * 6)
        num_rows = encoder_config.resolution[1] // encoder_config.block_size
        expected_i_frame_size = encoder_config.rc_lookup_table[encoder_config.quantization_factor]['I'] * num_rows
        expected_p_frame_size = encoder_config.rc_lookup_table[encoder_config.quantization_factor]['P'] * num_rows
        logger.info(f"overage [{num_rows}] : {frame_bits_consumed} | {frame_bits_consumed//num_rows} @ {encoder_config.quantization_factor} -> {round(frame_bits_consumed / expected_i_frame_size, 2)} | {round(frame_bits_consumed / expected_p_frame_size, 2)}")
        return frame_bits_consumed/expected_i_frame_size, frame_bits_consumed/expected_p_frame_size
    def is_iframe(self):
        return self.prediction_mode == PredictionMode.INTRA_FRAME
    def is_pframe(self):
        return self.prediction_mode == PredictionMode.INTER_FRAME

def apply_dct_and_quantization(residual_block, block_size, quantization_factor):
    dct_coffs = apply_dct_2d(residual_block)
    Q = generate_quantization_matrix(block_size, quantization_factor)
    quantized_dct_coffs = quantize_block(dct_coffs, Q)
    return quantized_dct_coffs, Q


def reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc):
    rescaled_dct_coffs = rescale_block(quantized_dct_coffs, Q)
    idct_residual_block = apply_idct_2d(rescaled_dct_coffs)
    reconstructed_block_with_mc = np.round(idct_residual_block + predicted_block_with_mc).astype(np.int16)
    clipped_reconstructed_block = np.clip(reconstructed_block_with_mc, 0, 255).astype(np.uint8)
    return clipped_reconstructed_block, idct_residual_block
