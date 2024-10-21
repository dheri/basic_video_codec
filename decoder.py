from contextlib import ExitStack

import numpy as np

from common import get_logger
from encoder.IFrame import IFrame
from encoder.PFrame import decode_p_frame, PFrame
from file_io import write_y_only_frame, FileIOHelper
from input_parameters import InputParameters
from motion_vector import parse_mv
from skimage.metrics import peak_signal_noise_ratio

logger = get_logger()


def decode(params: InputParameters):
    file_io = FileIOHelper(params)
    frames_to_process = params.frames_to_process
    height = params.height
    width = params.width
    mv_txt_file = file_io.get_mv_file_name()
    decoded_yuv = file_io.get_mc_decoded_file_name()

    frame_size = width * height
    prev_frame = np.full((height, width), 128, dtype=np.uint8)


    with ExitStack() as stack:
        quant_dct_coff_fh = stack.enter_context(open(file_io.get_quant_dct_coff_fh_file_name(), 'rb'))
        reconstructed_file_fh = stack.enter_context(open(file_io.get_mc_reconstructed_file_name(), 'rb'))
        mv_txt_fh = stack.enter_context(open(mv_txt_file, 'rt'))
        decoded_fh = stack.enter_context(open(decoded_yuv, 'wb'))

        frame_index = 0
        while True:
            frame_index += 1
            quant_dct_coff = quant_dct_coff_fh.read(frame_size*2) # quant_dct_coff are stored as int16. i.e. 2bytes
            mv_txt =  mv_txt_fh.readline()
            if not quant_dct_coff or frame_index > frames_to_process or not mv_txt:
                break  # End of file or end of frames
            logger.debug(f"Decoding frame {frame_index}/{frames_to_process}")
            quant_dct_coff_frame = np.frombuffer(quant_dct_coff, dtype=np.int16)
            quant_dct_coff_frame = quant_dct_coff_frame.reshape((height, width))

            mv = parse_mv(mv_txt)

            if False and frame_index % params.encoder_config.I_Period == 0:
                frame = IFrame()
            else:
                frame = PFrame()
                frame.quat_dct_coffs_frame_with_mc = quant_dct_coff_frame

            decoded_frame = decode_p_frame(quant_dct_coff_frame, prev_frame, mv, params.encoder_config)

            reconstructed_frame= np.frombuffer(reconstructed_file_fh.read(frame_size), dtype=np.uint8).reshape((height, width))
            psnr = peak_signal_noise_ratio(decoded_frame, reconstructed_frame)

            logger.info(f"{frame_index:2}: psnr [{round(psnr,2):6.2f}]")


            write_y_only_frame(decoded_fh, decoded_frame)

            prev_frame = decoded_frame
    print('end decoding')