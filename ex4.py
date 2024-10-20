from decoder import decode
from encoder.encoder import encode
from input_parameters import InputParameters
from metrics import plot_metrics


def main(params: InputParameters):
    encode(params)
    plot_metrics(params)
    decode(params)