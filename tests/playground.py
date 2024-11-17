import numpy as np
from bitarray import bitarray

from encoder.block_predictor import get_ref_block_at_mv, build_pre_interpolated_buffer
from encoder.params import EncoderConfig, logger


def map_to_non_negative(value):
    """
    Maps a signed integer to a non-negative integer suitable for Exp-Golomb encoding.
    """
    return 2 * value if value >= 0 else -2 * value - 1


def exp_golomb_encode(value):
    """
    Encodes a non-negative integer using Exp-Golomb encoding.
    """
    print(f"v: [{value}]")

    if value < 0:
        raise ValueError("exp_golomb_encode expects a non-negative integer.")

    if value == 0:
        return bitarray('0')

    # Calculate the number of leading zeros (m)
    m = 0
    while (1 << (m + 1)) <= value + 1:
        m += 1

    # Calculate the prefix (m leading zeros followed by a '1')
    prefix = '0' * m + '1'

    # Calculate the suffix as the remainder after removing the highest power of 2
    suffix_value = value - ((1 << m) - 1)

    if suffix_value < 0:
        raise ValueError(f"Unexpected negative remainder in suffix calculation: {suffix_value}")

    suffix = format(suffix_value, f'0{m}b')  # Binary representation of the remainder, padded to m bits
    print(f"p [{prefix}] s: [{suffix}]")

    return bitarray(prefix + suffix)


def encode_prediction_info(prediction_array):
    """
    Encodes an array of signed prediction values using Exponential-Golomb encoding.
    """
    encoded_stream = bitarray()
    for value in prediction_array:
        # Map the signed integer to a non-negative integer for Exp-Golomb encoding
        mapped_value = map_to_non_negative(value)

        # Encode the mapped non-negative integer
        encoded_bits = exp_golomb_encode(mapped_value)

        # Append to the encoded bitstream
        encoded_stream.extend(encoded_bits)

    return encoded_stream.tobytes()



if __name__ == '__main__':
    rf = np.array([
        [ 25, 28, 29, 29,],
        [ 50, 57, 53, 53,],
        [ 44, 56, 64, 76,],
        [ 46, 54, 66, 74,],
    ])
    origin = (1, 1)
    # mv_x, mv_y = 1 ,0
    ec = EncoderConfig(2, 5, I_Period=8, quantization_factor=0, fracMeEnabled=True)
    irf = build_pre_interpolated_buffer(rf )

    logger.info(f"interpolated Reference f :\n{irf}")
    logger.info(f"interpolated Reference Block\n"
                f"{get_ref_block_at_mv(rf, irf, origin, 1, 2, ec)}")