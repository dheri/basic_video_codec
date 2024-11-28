from decoder import decode_video
from encoder.encoder import encode_video
from input_parameters import InputParameters
from metrics.metrics import plot_metrics

# Part 1
# Multiple Reference Frames
def main(params: InputParameters):
    encode_video(params)
    plot_metrics(params)
    decode_video(params)
