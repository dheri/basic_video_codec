import numpy as np

def exp_golomb_encode(value):
    if value == 0:
        return '0'
    if value > 0:
        mapped_value = 2 * value - 1
    else:
        mapped_value = -2 * value
    m = int(np.log2(mapped_value + 1))
    prefix = '0' * m + '1'
    suffix = format(mapped_value - (1 << m), f'0{m}b')
    return prefix + suffix


def exp_golomb_decode(bitstream):
    m = 0
    while bitstream[m] == '0':
        m += 1
    value = (1 << m) + int(bitstream[m + 1:], 2)
    return value

def rle_encode(coeffs):
    encoded = []
    i = 0
    while i < len(coeffs):
        if coeffs[i] == 0:
            zero_count = 0
            while i < len(coeffs) and coeffs[i] == 0:
                zero_count += 1
                i += 1
            encoded.append(zero_count)  # Positive for run of zeros
        else:
            nonzero_count = 0
            start_idx = i
            while i < len(coeffs) and coeffs[i] != 0:
                nonzero_count += 1
                i += 1
            encoded.append(-nonzero_count)  # Negative for run of non-zeros
            encoded.extend(coeffs[start_idx:i])
    encoded.append(0)  # End of block
    return encoded
