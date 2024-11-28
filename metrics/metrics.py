import csv

import numpy as np
from matplotlib import pyplot as plt

from encoder.FrameMetrics import FrameMetrics
from encoder.params import logger
from file_io import FileIOHelper
from input_parameters import InputParameters

def plot_metrics(params: InputParameters):
    file_io = FileIOHelper(params)
    logger.info(f" cwd: {file_io.get_file_name('')}")
    csv_file_name = file_io.get_metrics_csv_file_name()

    frame_numbers, avg_mae_values, psnr_values, frame_bytes = read_metrics_from_csv(csv_file_name)

    plot_mae_psnr_vs_frame(file_io, frame_numbers, avg_mae_values, psnr_values)

    plot_rate_distortion(file_io, params, frame_bytes, psnr_values, avg_mae_values)


def read_metrics_from_csv(csv_file_name: str):
    frame_numbers = []
    avg_mae_values = []
    psnr_values = []
    frame_bytes = []

    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)  # Skip the header row

        for row in csv_reader:
            metrics = FrameMetrics.from_csv_row(row)
            frame_numbers.append(int(metrics.idx))
            avg_mae_values.append(float(metrics.avg_mae))
            psnr_values.append(float(metrics.psnr))
            frame_bytes.append(float(metrics.frame_bytes))

    return frame_numbers, avg_mae_values, psnr_values, frame_bytes


def plot_mae_psnr_vs_frame(file_io, frame_numbers, avg_mae_values, psnr_values):
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


def plot_rate_distortion(file_io, params, frame_bytes, psnr_values, avg_mae_values):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    frame_bytes = np.array(frame_bytes)
    psnr_values = np.array(psnr_values)
    avg_mae_values = np.array(avg_mae_values)

    # Sort data by frame_bytes
    sorted_indices = np.argsort(frame_bytes)
    frame_bytes_sorted = frame_bytes[sorted_indices]
    psnr_values_sorted = psnr_values[sorted_indices]
    mae_values_sorted = avg_mae_values[sorted_indices]
    best_fit_line_order = 3

    # Primary Y-axis: PSNR
    ax1.set_xlabel("Encoded frame size in Bytes")
    ax1.set_ylabel("PSNR (dB)", color="r")
    ax1.scatter(frame_bytes_sorted, psnr_values_sorted, marker='x', color='r', label='PSNR')
    best_fit_psnr = np.poly1d(np.polyfit(frame_bytes_sorted, psnr_values_sorted, best_fit_line_order))
    ax1.plot(frame_bytes_sorted, best_fit_psnr(frame_bytes_sorted), linestyle='dotted', linewidth=1, color='r')
    ax1.tick_params(axis='y', labelcolor="r")
    ax1.grid(True)

    # Secondary Y-axis: MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel("MAE", color="b")
    ax2.scatter(frame_bytes_sorted, mae_values_sorted, marker='o', linestyle='dashed', color='b', label='MAE')
    best_fit_mae = np.poly1d(np.polyfit(frame_bytes_sorted, mae_values_sorted, best_fit_line_order))
    ax2.plot(frame_bytes_sorted, best_fit_mae(frame_bytes_sorted), linestyle='dotted', linewidth=1, color='b')
    ax2.tick_params(axis='y', labelcolor="b")

    # Title and Legend
    ec = params.encoder_config
    fig.suptitle(f"RD Curve with PSNR over Frame size\n"
                 f"i [{ec.block_size}] r [{ec.search_range}] q [{ec.quantization_factor}] FracME [{ec.fracMeEnabled}] FastME [{ec.fastME}]")
    fig.tight_layout()
    plt.savefig(file_io.get_file_name("rd.png"))
    plt.close()
