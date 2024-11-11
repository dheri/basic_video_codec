from bitarray import bitarray


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


# Example usage:
prediction_array = [3, -2, 1, 0, -1]  # Sample prediction values (signed integers)
encoded_bytes = encode_prediction_info(prediction_array)
print(encoded_bytes)  # This is the encoded byte array
