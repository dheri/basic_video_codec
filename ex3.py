import logging
import os

from decoder import decode_video
from encoder.encoder import encode_video
from file_io import FileIOHelper
from input_parameters import InputParameters
from metrics import plot_metrics



def main(params: InputParameters):
    encode_video(params)
    plot_metrics(params)
    decode_video(params)


