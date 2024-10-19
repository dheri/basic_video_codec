import logging

from decoder import decode
from encoder import encode
from input_parameters import InputParameters
from metrics import plot_metrics

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(params: InputParameters):
    encode(params)
    # plot_metrics(params)
    # decode(params)