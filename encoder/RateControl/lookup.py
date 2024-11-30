import csv
import os

import pandas as pd

from common import get_logger
from encoder.FrameMetrics import FrameMetrics
from encoder.params import EncoderConfig
from input_parameters import InputParameters
from metrics.plot_rd_curves import create_label

def rc_lookup_file_path(ec: EncoderConfig, i_period_str=None):
    curr_dir = os.path.dirname(__file__)
    res_str = f"{ec.resolution[0]}_{ec.resolution[1]}"
    if not i_period_str:
        i_period_str = "I" if ec.I_Period == 1 else "P"
    output_file_name = os.path.join(curr_dir, f'lookups/{res_str}_{ec.block_size}_{i_period_str}.csv')
    return output_file_name
def generate_rc_lookup(metric_files, params: InputParameters):
    output_file_name = rc_lookup_file_path(params.encoder_config)
    i_period_str = "P"
    if params.encoder_config.I_Period == 1:
        i_period_str = "I"

    aggregated_results = {}

    for file_path in metric_files:
        # Extract block size and search range
        _, details = create_label(file_path)
        block_size = details["block_size"]
        search_range = params.encoder_config.search_range
        # Read metric file and compute aggregates
        with open(file_path, 'rt') as input_file:
            csv_reader = csv.reader(input_file)
            headers = next(csv_reader)  # Skip header row

            for row in csv_reader:
                metrics = FrameMetrics.from_csv_row(row)
                qp = details['qp']  # Assuming QP is part of FrameMetrics
                frame_bits = metrics.frame_bytes * 8
                blocks_per_col = params.height // block_size

                if qp not in aggregated_results:
                    aggregated_results[qp] = {
                        "I-Frame Bits": 0,
                        "P-Frame Bits": 0,
                        "I-Frame Rows": 0,
                        "P-Frame Rows": 0
                    }

                if metrics.is_i_frame:
                    aggregated_results[qp]["I-Frame Bits"] += frame_bits
                    aggregated_results[qp]["I-Frame Rows"] += blocks_per_col
                else:
                    aggregated_results[qp]["P-Frame Bits"] += frame_bits
                    aggregated_results[qp]["P-Frame Rows"] += blocks_per_col

    # Prepare data for writing to file
    transposed_data = {}
    for qp, data in sorted(aggregated_results.items()):
        if i_period_str == "I":
            avg_i_frame_bits = (
                data["I-Frame Bits"] / data["I-Frame Rows"] if data["I-Frame Rows"] > 0 else 0
            )
            transposed_data[qp] = [round(avg_i_frame_bits)]
        elif i_period_str == "P":
            avg_p_frame_bits = (
                data["P-Frame Bits"] / data["P-Frame Rows"] if data["P-Frame Rows"] > 0 else 0
            )
            transposed_data[qp] = [round(avg_p_frame_bits)]

    # Write transposed lookup table to file
    df = pd.DataFrame(transposed_data)
    df.to_csv(output_file_name, index=False)

    print(f"Transposed lookup table saved: {output_file_name}")


# def get_lookup_table_from_file(file_path: str):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"rc lookup file not found @ {file_path}")
#     df = pd.read_csv(file_path)
#     # Initialize the lookup table
#     lookup_table = {}
#
#     # Populate the lookup table
#     metrics = df.iloc[:, 0]  # The first column is the "Metric" (row names)
#     for column in df.columns[1:]:  # Skip the "Metric" column
#         qp = int(column)  # QP value (column name)
#         lookup_table[qp] = {metric: int(value) for metric, value in zip(metrics, df[column])}
#
#     return lookup_table


def get_combined_lookup_table(file_path_i: str, file_path_p: str):

    if not os.path.exists(file_path_i):
        raise FileNotFoundError(f"I-frame RC lookup file not found @ {file_path_i}")
    if not os.path.exists(file_path_p):
        raise FileNotFoundError(f"P-frame RC lookup file not found @ {file_path_p}")

    # Read I-frame lookup table
    df_i = pd.read_csv(file_path_i, header=None)  # No header
    lookup_table = {}

    # Process rows in I-frame file
    for qp, value in zip(df_i.iloc[0, 1:], df_i.iloc[1, 1:]):  # Skip row/column indices
        qp = int(qp)  # Convert QP value to int
        value = int(value)  # Convert bit rate to int
        if qp not in lookup_table:
            lookup_table[qp] = {}
        lookup_table[qp]["I"] = value

    # Read P-frame lookup table
    df_p = pd.read_csv(file_path_p, header=None)  # No header

    # Process rows in P-frame file
    for qp, value in zip(df_p.iloc[0, 1:], df_p.iloc[1, 1:]):  # Skip row/column indices
        qp = int(qp)  # Convert QP value to int
        value = int(value)  # Convert bit rate to int
        if qp not in lookup_table:
            lookup_table[qp] = {}
        lookup_table[qp]["P"] = value

    # Optionally calculate combined ('C') if needed
    for qp in lookup_table:
        i_value = lookup_table[qp].get("I", 0)
        p_value = lookup_table[qp].get("P", 0)
        lookup_table[qp]["C"] = (i_value + p_value) // 2

    return lookup_table