import csv
from contextlib import ExitStack

import numpy as np
from matplotlib import pyplot as plt

from encoder.FrameMetrics import FrameMetrics
from encoder.params import EncoderConfig, logger
from file_io import FileIOHelper
from input_parameters import InputParameters


def plot_metrics(params: InputParameters):
    file_io = FileIOHelper(params)
    logger.info(f" cwd: {file_io.get_file_name('')}")
    csv_file_name = file_io.get_metrics_csv_file_name()

    frame_numbers = []
    avg_mae_values = []
    psnr_values = []
    frame_bytes = []

    # Read the CSV file
    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)  # Skip the header row

        # Parse the rows into respective lists
        for row in csv_reader:
            idx, _, avg_mae, _, psnr,frame_size, _  = row
            frame_numbers.append(int(idx))           # Frame index
            avg_mae_values.append(float(avg_mae))   # Average MAE
            psnr_values.append(float(psnr))         # PSNR
            frame_bytes.append(float(frame_size)) # Total file bits

    # Plot 1: MAE and PSNR vs Frame Numbers
    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')
    plt.plot(frame_numbers, psnr_values, marker='x', linestyle='--', color='r', label='PSNR')
    plt.title("MAE and PSNR per Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Metric Value")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_io.get_file_name("mae_psnr_vs_frame.png"))
    plt.close()

    # Plot 2: Rate-Distortion with Dual Y-Scales
    fig, ax1 = plt.subplots(figsize=(10, 6))

    frame_bytes = np.array(frame_bytes)
    psnr_values = np.array(psnr_values)
    avg_mae_values = np.array(avg_mae_values)

    # Sort data by frame_bytes
    sorted_indices = np.argsort(frame_bytes)  # Get sorted indices
    frame_bytes_sorted = frame_bytes[sorted_indices]
    psnr_values_sorted = psnr_values[sorted_indices]
    mae_values_sorted = avg_mae_values[sorted_indices]
    best_fit_line_order = 3


    # Primary Y-axis: PSNR
    ax1.set_xlabel("Encoded frame size in Bytes")
    ax1.set_ylabel("PSNR (dB)", color="r")
    ax1.scatter(frame_bytes_sorted, psnr_values_sorted, marker='x', color='r', label='PSNR')
    best_fit_psnr = np.poly1d(np.polyfit(frame_bytes_sorted, psnr_values_sorted, best_fit_line_order))

    ax1.plot(frame_bytes_sorted, best_fit_psnr(frame_bytes_sorted), linestyle='dotted', linewidth=1, color='r', )

    ax1.tick_params(axis='y', labelcolor="r")
    ax1.grid(True)

    # Secondary Y-axis: MAE
    ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis
    ax2.set_ylabel("MAE", color="b")
    ax2.scatter(frame_bytes_sorted, mae_values_sorted, marker='o', linestyle='dashed', color='b', label='MAE')
    best_fit_mae = np.poly1d(np.polyfit(frame_bytes_sorted, mae_values_sorted, best_fit_line_order))

    ax2.plot(frame_bytes_sorted, best_fit_mae(frame_bytes_sorted), marker='o', linestyle='dotted', linewidth=1, color='b', label='MAE')
    ax2.tick_params(axis='y', labelcolor="b")

    # Title and Legend
    ec = params.encoder_config
    fig.suptitle(f"RD Curve with PSNR over Frame size\n"
                 f"i [{ec.block_size}] r [{ec.search_range}] q [{ec.quantization_factor}] FracME [{ec.fracMeEnabled}] FastME [{ec.fastME}]")
    fig.tight_layout()
    plt.savefig(file_io.get_file_name("rd.png"))
    plt.close()

def plot_combined_metrics(metric_files, seq_name):
    with ExitStack() as stack:
        file_handles = [stack.enter_context(open(f_name)) for f_name in metric_files]
        plt.figure(figsize=(12, 8))

        # Iterate through each file and plot its data
        for idx, f in enumerate(file_handles):
            csv_reader = csv.reader(f)
            headers = next(csv_reader)  # Skip the header row

            frame_bytes = []
            psnr_values = []

            # Parse the rows into respective lists
            for row in csv_reader:
                metrics = FrameMetrics.from_csv_row(row)

                frame_bytes.append(float(metrics.file_bits))
                psnr_values.append(float(metrics.psnr))

            # Sort data by frame_bytes to ensure proper line fitting
            frame_bytes = np.array(frame_bytes)
            psnr_values = np.array(psnr_values)
            sorted_indices = np.argsort(frame_bytes)
            frame_bytes_sorted = frame_bytes[sorted_indices]
            psnr_values_sorted = psnr_values[sorted_indices]

            # Plot data for this file as a separate series
            plt.scatter(frame_bytes_sorted, psnr_values_sorted, marker='x', label=f" {seq_name} {create_label(metric_files[idx])[0]}", alpha=0.4)
            # Add a best-fit line (e.g., polynomial order 2)
            best_fit_line = np.poly1d(np.polyfit(frame_bytes_sorted, psnr_values_sorted, 2))
            plt.plot(frame_bytes_sorted, best_fit_line(frame_bytes_sorted), linestyle='dotted', linewidth=0.5)

        # Add labels, title, legend, and grid
        plt.xlabel("bits in file")
        plt.ylabel("PSNR (dB)")
        plt.title("Combined Frame Bytes vs PSNR for Multiple Files")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(f"../data/assign2_dels/{seq_name}.png")
        # plt.show()
        plt.close()



def create_label(file_path):
    parts = file_path.split('/')

    if len(parts) < 4:
        raise ValueError("File path does not match the expected format.")

    # Extract sequence name and encoder configuration
    seq_name = parts[2]  # Second directory is the sequence name
    encoder_config = parts[3]  # Third directory is the encoder config

    # Parse encoder configuration
    try:
        block_size, search_range, qp, nRefFrames = encoder_config.split('_')
        block_size = int(block_size)
        search_range = float(search_range)
        qp = int(qp)
        nRefFrames = int(nRefFrames)

        # Determine fracMe and fastME based on search_range
        fracMeEnabled = '.' in str(search_range)
        fastMeEnabled = search_range < 0
    except ValueError:
        raise ValueError(
            "Encoder configuration does not match the expected format: block_size_search_range_qp_nRefFrame.")

    # Create the label string
    label = f"qp={qp}"

    # Preserve parsed details in a dictionary
    details = {
        "seq_name": seq_name,
        "block_size": block_size,
        "search_range": search_range,
        "qp": qp,
        "nRefFrames": nRefFrames,
        "fracMeEnabled": fracMeEnabled,
        "fastMeEnabled": fastMeEnabled,
    }

    return label, details


