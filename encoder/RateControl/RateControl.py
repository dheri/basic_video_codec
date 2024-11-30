from encoder.params import EncoderConfig


def bit_budget_per_frame(ec):
    if not ec.RCflag:
        raise ValueError("Rate Control is OFF. Bit budget calculation is only relevant when RC is ON.")
    return ec.targetBR / ec.frame_rate


def calculate_row_bit_budget(remaining_bits, row_idx, ec: EncoderConfig):

    frame_width, frame_height = ec.resolution

    blocks_per_row = frame_width // ec.block_size
    blocks_per_col = frame_height // ec.block_size

    total_rows_of_blocks = blocks_per_col - row_idx

    # Bit budget per row of blocks
    row_bit_budget = remaining_bits / total_rows_of_blocks

    return row_bit_budget


def find_rc_qp_for_row(bit_budget, qp_table, frame_type="C"):
    if frame_type not in ['I', 'P', 'C']:
        raise ValueError("Invalid frame type. Must be one of 'I', 'P', or 'C'.")

    for qp, bits in sorted(qp_table.items()):
        if bits[frame_type] <= bit_budget:
            return qp
    return max(qp_table.keys())
