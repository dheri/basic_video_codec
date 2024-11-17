from decoder import decode_video
from encoder.encoder import encode_video
from input_parameters import InputParameters
from metrics import plot_metrics

def main(params: InputParameters):
    encode_video(params)  
    #plot_metrics(params)  # Plot R-D performance or other metrics
    decode_video(params)  # Perform decoding to verify results
