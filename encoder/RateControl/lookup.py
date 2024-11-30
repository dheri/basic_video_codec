import csv
import os

import pandas as pd

from encoder.FrameMetrics import FrameMetrics
from encoder.params import EncoderConfig
from input_parameters import InputParameters
from metrics.plot_rd_curves import create_label

def rc_lookup_file_path(ec: EncoderConfig):
    curr_dir = os.path.dirname(__file__)
    output_file_name = os.path.join(curr_dir, f'lookups/{ec.block_size}.csv')
    return output_file_name

def generate_rc_lookup(metric_files, params: InputParameters):
    output_file_name = rc_lookup_file_path(params.encoder_config)
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
                frame_bits = metrics.frame_bytes
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
    transposed_data = {
        "Metric": ["I", "P", "C"]
    }

    for qp, data in sorted(aggregated_results.items()):
        avg_i_frame_bits = (
            data["I-Frame Bits"] / data["I-Frame Rows"] if data["I-Frame Rows"] > 0 else 0
        )
        avg_p_frame_bits = (
            data["P-Frame Bits"] / data["P-Frame Rows"] if data["P-Frame Rows"] > 0 else 0
        )
        avg_combined_bits = (
            (data["I-Frame Bits"] + data["P-Frame Bits"]) /
            (data["I-Frame Rows"] + data["P-Frame Rows"])
            if (data["I-Frame Rows"] + data["P-Frame Rows"]) > 0 else 0
        )
        transposed_data[qp] = [round(avg_i_frame_bits), round(avg_p_frame_bits), round(avg_combined_bits)]

    # Write transposed lookup table to file
    df = pd.DataFrame(transposed_data)
    df.to_csv(output_file_name, index=False)

    print(f"Transposed lookup table saved: {output_file_name}")

def get_lookup_table_from_file(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"rc lookup file not found @ {file_path}")
    df = pd.read_csv(file_path)
    # Initialize the lookup table
    lookup_table = {}

    # Populate the lookup table
    metrics = df.iloc[:, 0]  # The first column is the "Metric" (row names)
    for column in df.columns[1:]:  # Skip the "Metric" column
        qp = int(column)  # QP value (column name)
        lookup_table[qp] = {metric: int(value) for metric, value in zip(metrics, df[column])}

    return lookup_table

