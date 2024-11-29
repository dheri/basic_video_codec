import csv
import os
from typing import LiteralString

import pandas as pd

from common import get_logger
from encoder.FrameMetrics import FrameMetrics
from input_parameters import InputParameters
from metrics.plot_rd_curves import create_label
logger = get_logger()

def print_average_bit_count_per_block_row(metric_files, params:InputParameters):


    results = {}

    for file_path in metric_files:
        # Use create_label to extract the block size
        _, details = create_label(file_path)
        block_size = details["block_size"]
        # Initialize aggregates
        p_frame_bits = 0
        i_frame_bits = 0
        p_frame_rows = 0
        i_frame_rows = 0


        bn = os.path.basename(file_path).replace(".csv", "_avg_bits.csv")
        output_file_name = os.path.join(os.path.dirname(file_path), f"{bn}")
        # Read CSV data
        with open(file_path, 'rt') as input_file, open(output_file_name, 'wt', newline='') as output_file:
            logger.debug(f" {output_file_name}, {file_path}")
            csv_reader = csv.reader(input_file)
            csv_writer = csv.writer(output_file)
            headers = next(csv_reader)  # Skip header row

            total_bits = 0
            total_block_rows = 0

            for row in csv_reader:
                metrics = FrameMetrics.from_csv_row(row)

                # Calculate number of rows of blocks
                blocks_per_row = params.width // block_size
                blocks_per_col = params.height // block_size

                frame_bits = metrics.frame_bytes
                total_block_rows += blocks_per_col
                if metrics.is_i_frame:
                    i_frame_bits += frame_bits
                    i_frame_rows += blocks_per_col
                else:
                    p_frame_bits += frame_bits
                    p_frame_rows += blocks_per_col


                total_bits += frame_bits
                avg_bits_per_row_frame = frame_bits / blocks_per_col if blocks_per_col > 0 else 0

                csv_writer.writerow([metrics.idx, round(avg_bits_per_row_frame, 2)])

            # Calculate average bits per row of blocks
            avg_p_bits_per_row = p_frame_bits / p_frame_rows if p_frame_rows > 0 else 0
            avg_i_bits_per_row = i_frame_bits / i_frame_rows if i_frame_rows > 0 else 0
            avg_bits_per_row = total_bits / total_block_rows if total_block_rows > 0 else 0
            # Store the results
            results[file_path] = {
                "Block Size": block_size,
                "Avg I-Frame Bits/Row": f"{avg_i_bits_per_row:.2f}",
                "Avg P-Frame Bits/Row": f"{avg_p_bits_per_row:.2f}",
                "Avg   Frame Bits/Row": f"{avg_bits_per_row:.2f}"
            }
            # print(f" avg_i_bits_per_row = {i_frame_bits} / {i_frame_rows} ")
            # print(f" avg_p_bits_per_row = {p_frame_bits} / {p_frame_rows} ")
            logger.info(results[file_path])

    # Display results as a table
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'File'}, inplace=True)
    # logger.info(results_df)
