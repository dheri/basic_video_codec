from contextlib import ExitStack

from encoder.IFrame import IFrame
from encoder.PFrame import  PFrame
from encoder.PredictionMode import PredictionMode
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from skimage.metrics import peak_signal_noise_ratio
from encoder.entropy_encoder import *
logger = get_logger()

def entropy_decode(bitstream):
    decoded_symbols = []
    while len(bitstream) > 0:
        symbol, bitstream = exp_golomb_decode(bitstream)
        decoded_symbols.append(symbol)
    return decoded_symbols


def decode_video(params: InputParameters):
    file_io = FileIOHelper(params)
    frames_to_process = params.frames_to_process
    height = params.height
    width = params.width
    decoded_yuv = file_io.get_mc_decoded_file_name()

    frame_size = width * height
    prev_frame = np.full((height, width), 128, dtype=np.uint8)


    with ExitStack() as stack:
        quant_dct_coff_fh = stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'rb'))
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        encoded_fh = stack.enter_context(open(file_io.get_encoded_file_name(), 'rb'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            quant_dct_coff = quant_dct_coff_fh.read(frame_size * 2)  # quant_dct_coff are stored as int16. i.e. 2 bytes
            if not quant_dct_coff or frame_index > frames_to_process or not encoded_fh:
                break  # End of file or end of frames
            prediction_mode = int.from_bytes(encoded_fh.read(1))
            logger.debug(f"Decoding {PredictionMode(prediction_mode)} {frame_index}/{frames_to_process}")

            if prediction_mode == PredictionMode.INTRA_FRAME.value:
                frame = IFrame()
            else:
                frame = PFrame(prev_frame=prev_frame)

            prediction_data_len = int.from_bytes(encoded_fh.read(2))
            prediction_data = encoded_fh.read(prediction_data_len)
            frame.entropy_decode_prediction_data(prediction_data, params)

            ee_dct_coffs_len = int.from_bytes(encoded_fh.read(3))
            frame.entropy_encoded_DCT_coffs = encoded_fh.read(ee_dct_coffs_len)
            frame.entropy_decode_dct_coffs(params)


            # Decode the frame
            decoded_frame = frame.decode_mc_q_dct((params.height, params.width), encoder_config=params.encoder_config)

            # Read and compare with the reconstructed frame
            reconstructed_frame = np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape((height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)
            dct_coffs_extremes = frame.get_quat_dct_coffs_extremes()

            logger.info(
                f"{frame_index:2}: psnr [{round(psnr, 2):6.2f}], q_dct_range: [{dct_coffs_extremes[0]:4}, {dct_coffs_extremes[1]:3}]")

            # Write the decoded frame to the output file
            write_y_only_frame(decoded_fh, decoded_frame)

            # Update the previous frame
            prev_frame = decoded_frame

    logger.info('End decoding')