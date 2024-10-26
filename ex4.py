from decoder import decode_video
from encoder.encoder import encode_video
from input_parameters import InputParameters
from metrics import plot_metrics

def golomb_encoding(val):
    if val == 0:
        return '1'
    sign = 0 if val >= 0 else 1
    val = abs(val)
    binary_val = bin(val)[2:]
    num_prefix_zeros = len(binary_val) - 1
    prefix = '0' * num_prefix_zeros
    return prefix + binary_val + str(sign)

def main(params: InputParameters):
    encode_video(params)
    # plot_metrics(params)
    decode_video(params)