import csv
import os
from typing import LiteralString

import pandas as pd

from encoder.FrameMetrics import FrameMetrics
from input_parameters import InputParameters
from metrics.plot_rd_curves import create_label


def print_average_bit_count_per_block_row(metric_files, params:InputParameters):


    results = []

    for file_path in metric_files:
        # Use create_label to extract the block size
        _, details = create_label(file_path)
        block_size = details["block_size"]

        bn = os.path.basename(file_path).replace(".csv", "_avg_bits.csv")
        output_file_name = os.path.join(os.path.dirname(file_path), f"{bn}")
        # Read CSV data
        with open(file_path, 'rt') as input_file, open(output_file_name, 'wt', newline='') as output_file:
            print(f" {output_file_name}, {file_path}")
            csv_reader = csv.reader(input_file)
            csv_writer = csv.writer(output_file)
            headers = next(csv_reader)  # Skip header row

            total_bits = 0
            total_block_rows = 0

            for row in csv_reader:
                metrics = FrameMetrics.from_csv_row(row)

                # Assume frame width and height are part of FrameMetrics (add attributes if needed)
                frame_width = params.width
                frame_height = params.height
                frame_bits = metrics.file_bits

                # Calculate number of rows of blocks
                blocks_per_row = frame_width // block_size
                blocks_per_col = frame_height // block_size

                # Total rows of blocks
                total_block_rows += blocks_per_col
                total_bits += frame_bits
                avg_bits_per_row = frame_bits / total_block_rows if total_block_rows > 0 else 0

                csv_writer.writerow([metrics.idx, avg_bits_per_row])


            # Calculate average bits per row of blocks
            avg_bits_per_row = total_bits / total_block_rows if total_block_rows > 0 else 0
            results.append({
                "File": file_path,
                "Block Size": block_size,
                "Avg Bits/Row": avg_bits_per_row
            })

    # Display results as a table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
