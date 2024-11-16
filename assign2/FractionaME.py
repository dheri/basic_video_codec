from decoder import decode_video
from encoder.encoder import encode_video
from input_parameters import InputParameters
from metrics import plot_metrics

# Part 4
# Fast ME
def main(params: InputParameters):
    encode_video(params)
    # plot_metrics(params)
    decode_video(params)
