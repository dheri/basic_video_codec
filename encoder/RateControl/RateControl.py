from common import get_logger
from encoder.Frame import Frame
from encoder.params import EncoderConfig

logger = get_logger()
def bit_budget_per_frame(ec):
    return ec.targetBR / ec.frame_rate


def calculate_constant_row_bit_budget(remaining_bits, row_idx, ec: EncoderConfig):
    frame_width, frame_height = ec.resolution

    blocks_per_row = frame_width // ec.block_size
    blocks_per_col = frame_height // ec.block_size

    total_rows_of_blocks = blocks_per_col - row_idx

    # Bit budget per row of blocks
    row_bit_budget = remaining_bits / total_rows_of_blocks

    return row_bit_budget


def calculate_proportional_row_bit_budget(frame: Frame, row_idx, ec: EncoderConfig):
    prev_pass_frame: Frame = frame.prev_pass_frame
    if prev_pass_frame is None:
        raise ValueError("cant find proportional bit budget as prev_pass_frame not defined")
    bit_usage_proportion = prev_pass_frame.bits_per_row[row_idx] /  sum(prev_pass_frame.bits_per_row)
    row_bit_budget = bit_budget_per_frame(ec) * bit_usage_proportion
    logger.debug(f"proportional_row_bit_budget[{row_idx:2d}] {100*bit_usage_proportion:3.2f}% is {row_bit_budget:6.0f}")
    return row_bit_budget



def find_rc_qp_for_row(bit_budget, qp_table, frame_type="C"):
    if frame_type not in ['I', 'P', 'C']:
        raise ValueError("Invalid frame type. Must be one of 'I', 'P', or 'C'.")

    for qp, bits in sorted(qp_table.items()):
        if bits[frame_type] <= bit_budget:
            # logger.debug(f"qp = {qp}")
            return qp
    return max(qp_table.keys())
