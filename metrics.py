import csv
from contextlib import ExitStack
from copy import copy

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
            metrics = FrameMetrics.from_csv_row(row)

            # idx, _, avg_mae, _, psnr,frame_size, _  = row
            frame_numbers.append(int(metrics.idx))           # Frame index
            avg_mae_values.append(float(metrics.avg_mae))   # Average MAE
            psnr_values.append(float(metrics.psnr))         # PSNR
            frame_bytes.append(float(metrics.frame_bytes)) # Total file bits

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


def create_label(file_path):
    parts = file_path.split('/')

    if len(parts) < 4:
        raise ValueError("File path does not match the expected format.")

    # Extract sequence name and encoder configuration
    file_name = parts[2]  # Second directory is the sequence name
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
        "file_name": file_name,
        "block_size": block_size,
        "search_range": search_range,
        "qp": qp,
        "nRefFrames": nRefFrames,
        "fracMeEnabled": fracMeEnabled,
        "fastMeEnabled": fastMeEnabled,
    }

    return label, details

def plot_overlay_metrics(base_metric_files, metric_files, seq_name):
    """
    Creates a graph with scatter and best-fit lines for both base_metric_files and metric_files.

    Parameters:
        base_metric_files (list): List of base metric CSV file paths.
        metric_files (list): List of additional metric CSV file paths.
        seq_name (str): Name of the sequence for labeling the graph.

    Saves:
        PNG file with the graph plotted.
    """
    plt.close('all')  # Close all existing figures to start fresh
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axes

    # Define a color map to maintain consistent colors for sequences
    color_map = {}
    color_palette = plt.cm.tab10.colors  # Use a tab10 color palette
    color_index = 0

    def get_color(label):
        nonlocal color_index
        if label not in color_map:
            color_map[label] = color_palette[color_index % len(color_palette)]
            color_index += 1
        return color_map[label]

    def process_files(file_list, label_prefix, is_base=False):
        with ExitStack() as stack:
            file_handles = [stack.enter_context(open(f_name)) for f_name in file_list]

            # Iterate through each file and plot its data
            for idx, f in enumerate(file_handles):
                csv_reader = csv.reader(f)
                headers = next(csv_reader)  # Skip the header row

                file_bits = []
                psnr_values = []
                encoding_time = 0

                # Parse the rows into respective lists
                for row in csv_reader:
                    metrics = FrameMetrics.from_csv_row(row)
                    file_bits.append(float(metrics.file_bits))
                    psnr_values.append(float(metrics.psnr))
                    encoding_time = metrics.elapsed_time

                # Sort data by file_bits to ensure proper line fitting
                file_bits = np.array(file_bits)
                psnr_values = np.array(psnr_values)
                sorted_indices = np.argsort(file_bits)
                frame_bytes_sorted = file_bits[sorted_indices]
                psnr_values_sorted = psnr_values[sorted_indices]

                # Generate a label for the sequence
                seq_label = f"{label_prefix} \t {create_label(file_list[idx])[0]} \t t={encoding_time:.2f}s"
                color = get_color(label_prefix)  # Get consistent color for this label

                # Plot scatter and best-fit line
                marker = '.' if is_base else 'x'
                ax.scatter(frame_bytes_sorted, psnr_values_sorted, marker=marker,  label=seq_label, color=color, alpha=0.8)
                best_fit_line = np.poly1d(np.polyfit(frame_bytes_sorted, psnr_values_sorted, 2))
                l_width = 0.5 if is_base else 0.7
                ax.plot(frame_bytes_sorted, best_fit_line(frame_bytes_sorted), linestyle='dotted', linewidth=l_width, color=color, alpha=0.6)

    # Process base_metric_files and metric_files
    process_files(base_metric_files, "assign1", is_base=True)
    process_files(metric_files, seq_name)

    # Add labels, title, legend, and grid
    ax.set_xlabel("Bits in File")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSRN vs File_Bits |  assign_1 vs  [{seq_name}]")

    ax.legend(loc='lower right')
    ax.grid(True)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(f"../data/assign2_dels/{seq_name}.png")
    plt.close(fig)
