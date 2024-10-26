import concurrent

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from typing import Self

from common import get_logger, generate_residual_block, find_mv_predicted_block, signed_to_unsigned, unsigned_to_signed
from encoder.Frame import Frame
from encoder.PredictionMode import PredictionMode
from encoder.block_predictor import predict_block
from encoder.dct import apply_dct_2d, generate_quantization_matrix, quantize_block, rescale_block, apply_idct_2d
from encoder.params import EncoderConfig, EncodedBlock
from concurrent import futures

from file_io import write_mv_to_file, write_y_only_frame

logger = get_logger()



class PFrame(Frame):
    def __init__(self, curr_frame=None, prev_frame=None):
        super().__init__(curr_frame, prev_frame)
        self.prediction_mode = PredictionMode.INTER_FRAME
        self.mv_field = None
        self.avg_mae = None

    def encode(self, encoder_config: EncoderConfig) -> Self :
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
        quat_dct_coffs_frame_with_mc = np.zeros_like(self.curr_frame, dtype=np.int16)

        # Process blocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futuress = [executor.submit(self.process_block,  x, y, block_size, search_range, quantization_factor, width, height, mv_field)
                       for y in range(0, height, block_size)
                       for x in range(0, width, block_size)]

            for f in concurrent.futures.as_completed(futuress):
                encoded_block = f.result()
                block_cords = encoded_block.block_coords
                x, y = block_cords

                # Update frames with the encoded block data
                reconstructed_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_block_with_mc
                residual_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.reconstructed_residual_block
                quat_dct_coffs_frame_with_mc[y:y + block_size, x:x + block_size] = encoded_block.quantized_dct_coffs

                mae_of_blocks += encoded_block.mae

        avg_mae = mae_of_blocks / num_of_blocks

        self.mv_field = mv_field  # Populate the motion vector field
        self.prediction_data = mv_field_to_bytearray(self.mv_field)  # Convert to byte array


        self.avg_mae = avg_mae
        self.residual_frame = residual_frame_with_mc
        self.quantized_dct_residual_frame = quat_dct_coffs_frame_with_mc
        self.reconstructed_frame = reconstructed_frame_with_mc
        return self


    def process_block(self, x, y, block_size, search_range, quantization_factor, width, height, mv_field):
        curr_block = self.curr_frame[y:y + block_size, x:x + block_size].astype(np.int16)

        # Get motion vector and MAE
        motion_vector, best_match_mae = self.get_motion_vector(curr_block, x, y, block_size, search_range, width, height)
        mv_field[(x, y)] = motion_vector

        # Generate residual and predicted block
        predicted_block_with_mc, residual_block_with_mc = generate_residual_block(curr_block, self.prev_frame, motion_vector, x, y, block_size)

        # Apply DCT and quantization
        quantized_dct_coffs, Q = apply_dct_and_quantization(residual_block_with_mc, block_size, quantization_factor)

        # Reconstruct the block using the predicted and inverse DCT
        clipped_reconstructed_block, idct_residual_block = reconstruct_block(quantized_dct_coffs, Q, predicted_block_with_mc)

        return EncodedBlock((x, y), motion_vector, best_match_mae, quantized_dct_coffs, idct_residual_block, clipped_reconstructed_block)


    def get_motion_vector(self, curr_block, x, y, block_size, search_range, width, height):
        prev_partial_frame_x_start_idx = max(x - search_range, 0)
        prev_partial_frame_x_end_idx = min(x + block_size + search_range, width)
        prev_partial_frame_y_start_idx = max(y - search_range, 0)
        prev_partial_frame_y_end_idx = min(y + block_size + search_range, height)

        prev_partial_frame = self.prev_frame[prev_partial_frame_y_start_idx:prev_partial_frame_y_end_idx,
                                        prev_partial_frame_x_start_idx:prev_partial_frame_x_end_idx]

        best_mv_within_search_window, best_match_mae, best_match_block = predict_block(curr_block, prev_partial_frame, block_size)

        motion_vector = [best_mv_within_search_window[0] + prev_partial_frame_x_start_idx - x,
                         best_mv_within_search_window[1] + prev_partial_frame_y_start_idx - y]

        return motion_vector, best_match_mae

    def decode(self, frame_size, encoder_config: EncoderConfig):
        return decode_p_frame(self.quantized_dct_residual_frame, self.prev_frame, self.mv_field, encoder_config)

    def parse_prediction_data(self, params):
        prediction_data = self.prediction_data
        # logger.info(f"parse prediction data: [{len(prediction_data)}] {prediction_data.hex()}")  # Log the byte array in hex format
        self.mv_field = byte_array_to_mv_field(self.prediction_data, params.width, params.height, params.encoder_config.block_size )  # Convert back to motion vector field

    def generate_prediction_data(self):
        # convert mv or inta-modes to byte array
        self.prediction_data = bytearray()  # Initialize an empty bytearray for prediction data

        # Iterate over the motion vector field
        for (i, j), (mv_x, mv_y) in self.mv_field.items():
            self.prediction_data.append(signed_to_unsigned(mv_x, 8))
            self.prediction_data.append(signed_to_unsigned(mv_y, 8))

        # logger.info(f"Generated prediction data: [{len(self.prediction_data)}] {self.prediction_data.hex()}")  # Log the byte array in hex format


def mv_field_to_bytearray(mv_field):
    byte_stream = bytearray()
    for mv in mv_field.values():
        for value in mv:
            # Convert signed integer to unsigned byte
            unsigned_byte = (value + 256) % 256  # Ensure the result is within 0-255
            byte_stream.append(unsigned_byte)
    return byte_stream

def byte_array_to_mv_field(byte_stream, width, height, block_size):
    mv_field = {}
    index = 0
    # Iterate through the byte stream in pairs
    for i in range(0, len(byte_stream), 2):
        # Ensure we have enough bytes left for mv_x and mv_y
        if i + 1 >= len(byte_stream):
            break

        mv_x_byte = byte_stream[i]
        mv_y_byte = byte_stream[i + 1]

        mv_x = unsigned_to_signed(mv_x_byte, 8)
        mv_y = unsigned_to_signed(mv_y_byte, 8)

        # Calculate the pixel coordinates (i, j) for the top-left corner of the block
        row_index = (index // (width // block_size)) * block_size  # Row (y-coordinate)
        column_index = (index % (width // block_size)) * block_size    # Column (x-coordinate)

        # Ensure the calculated (i, j) is within the frame dimensions
        if row_index < height and column_index < width:
            # Store the motion vector in the dictionary with the (i, j) as the key
            mv_field[(column_index, row_index)] = [mv_x, mv_y]  # Using tuple for fixed size
        else:
            print(f"Warning: Calculated coordinates {(column_index, row_index)} are out of bounds.")

        index += 1  # Increment the index for the next motion vector

    return mv_field



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


def decode_p_frame(quant_dct_coff_frame, prev_frame, mv_field, encoder_config: EncoderConfig):
    block_size = encoder_config.block_size
    quantization_factor = encoder_config.quantization_factor
    height, width = quant_dct_coff_frame.shape
    decoded_frame = np.zeros_like(prev_frame, dtype=np.uint8)

    # Generate the quantization matrix Q based on block size and quantization factor
    Q = generate_quantization_matrix(block_size, quantization_factor)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # print(f" [{x}, {y}]")
            # Get the quantized residual block
            dct_coffs_block = quant_dct_coff_frame[y:y + block_size, x:x + block_size]

            # Rescale the residual block by multiplying by Q
            rescaled_dct_coffs_block = rescale_block(dct_coffs_block, Q)

            # Apply inverse DCT to the rescaled residual block
            idct_residual_block = apply_idct_2d(rescaled_dct_coffs_block)

            # Get the predicted block using the motion vector
            predicted_b = find_mv_predicted_block(mv_field[(x, y)], x, y, prev_frame, block_size).astype(np.int16)

            # Reconstruct the block by adding the predicted block and the rescaled residual
            decoded_block = np.round(idct_residual_block + predicted_b).astype(np.int16)

            # Clip values to avoid overflow/underflow and convert back to uint8
            decoded_block = np.clip(decoded_block, 0, 255).astype(np.uint8)

            # Place the reconstructed block in the decoded frame
            decoded_frame[y:y + block_size, x:x + block_size] = decoded_block

    return decoded_frame
