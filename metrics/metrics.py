import csv

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from prettytable import PrettyTable

from encoder.FrameMetrics import FrameMetrics
from encoder.params import logger
from file_io import FileIOHelper
from input_parameters import InputParameters

def plot_metrics(params: InputParameters):
    file_io = FileIOHelper(params)
    logger.info(f" cwd: {file_io.get_file_name('')}")
    csv_file_name = file_io.get_metrics_csv_file_name()

    frame_numbers, is_i_frame_values, avg_mae_values, psnr_values, frame_bytes, file_bits = read_metrics_from_csv(csv_file_name)

    plot_series_vs_frame_numbers(file_io, frame_numbers, is_i_frame_values, avg_mae_values, 'Avg MAE', 'MAE per Frame')
    plot_series_vs_frame_numbers(file_io, frame_numbers, is_i_frame_values, psnr_values, 'PSNR', 'PSNR Frame')
    plot_series_vs_frame_numbers(file_io, frame_numbers, is_i_frame_values, np.array(frame_bytes) * 8, 'frame bits', 'Frame size')
    plot_series_vs_frame_numbers(file_io, file_bits, is_i_frame_values, psnr_values, 'rd curve', 'RD')
    plot_rd_v2(file_io, params, file_bits, frame_numbers, is_i_frame_values, psnr_values)
    # plot_psnr_vs_frame_bits_a(file_io, params, frame_numbers, frame_bytes, psnr_values)
    # plot_psnr_vs_frame_bits_b(file_io, frame_bytes, psnr_values)


def read_metrics_from_csv(csv_file_name: str):
    frame_numbers = []
    is_i_frame_values = []
    avg_mae_values = []
    psnr_values = []
    frame_bytes = []
    file_bits = []
    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)  # Skip the header row

        for row in csv_reader:
            metrics = FrameMetrics.from_csv_row(row)
            frame_numbers.append(int(metrics.idx))
            is_i_frame_values.append(int(metrics.is_i_frame))
            avg_mae_values.append(float(metrics.avg_mae))
            psnr_values.append(float(metrics.psnr))
            frame_bytes.append(float(metrics.frame_bytes))
            file_bits.append(float(metrics.file_bits))

    return frame_numbers, is_i_frame_values, avg_mae_values, psnr_values, frame_bytes, file_bits


def plot_series_vs_frame_numbers(file_io, frame_numbers, is_i_frame_values, series, series_name, plot_title):
    fig, ax1 = plt.subplots(figsize=(6, 4))


    plt.plot(frame_numbers, series, marker='o', linestyle='dotted', markersize=5, color='red', label=series_name, zorder=1, alpha=0.7)
    is_i_frame_values = np.array(is_i_frame_values) * np.array(series)
    is_i_frame_values = list(map(lambda  x: None if not x else x , is_i_frame_values))

    plt.scatter(frame_numbers, is_i_frame_values , marker='x', s=64,  color='black', label='I-Frame', zorder=2)

    plt.title(plot_title)
    plt.xlabel("Frame Number", )
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.ylabel(series_name)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_io.get_file_name(f"{series_name.lower().replace(' ',  '_')}.png"))
    plt.close('all')


def plot_rd_v2(file_io, params, file_bits, frame_numbers, is_i_frame_values, psnr_values):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    file_bits = np.array(file_bits)
    psnr_values = np.array(psnr_values)

    # Sort data by frame_bytes
    # sorted_indices = np.argsort(file_bits)
    frame_bytes_sorted = file_bits
    psnr_values_sorted = psnr_values
    best_fit_line_order =4


    is_i_frame_values = np.array(is_i_frame_values) * np.array(psnr_values)
    is_i_frame_values = list(map(lambda  x: None if not x else x , is_i_frame_values))
    plt.scatter(frame_bytes_sorted, is_i_frame_values , marker='x', s=64,  color='black', label='I-Frame', zorder=2)

    # Primary Y-axis: PSNR
    ax1.set_xlabel("Encoded frame size in Bytes")
    ax1.set_ylabel("PSNR (dB)", color="r")
    ax1.scatter(frame_bytes_sorted, psnr_values_sorted, marker='o', color='r', label='PSNR')
    best_fit_psnr = np.poly1d(np.polyfit(frame_bytes_sorted, psnr_values_sorted, best_fit_line_order))
    ax1.plot(frame_bytes_sorted, best_fit_psnr(frame_bytes_sorted), linestyle='dotted', linewidth=1, color='r')
    ax1.tick_params(axis='y', labelcolor="r")
    ax1.grid(True)

    # Title and Legend
    ec = params.encoder_config
    fig.suptitle(f"RD Curve with PSNR over File Bits\n"
                 f"i [{ec.block_size}] r [{ec.search_range if ec.search_range > 0 else '-'}] q [{ec.quantization_factor}] "
                 f"FracME [{ec.fracMeEnabled}] FastME [{ec.fastME}]"
                 f"")
    fig.tight_layout()
    plt.savefig(file_io.get_file_name("rdv2.png"))
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
def plot_psnr_vs_frame_bits_a(file_io, params, frame_numbers, frame_bytes, psnr_values):

    frame_bits = np.array(frame_bytes) * 8
    psnr_values = np.array(psnr_values)
    frame_numbers = np.array(frame_numbers)
    sorted_indices = np.argsort(frame_numbers)

    frame_bits_sorted = frame_bits[sorted_indices]
    psnr_values_sorted = psnr_values[sorted_indices]
    best_fit_line_order = 3

    # Create plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot PSNR
    ax1.set_xlabel("Frame Index", fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylabel("PSNR (dB)", color="r", fontsize=12)
    ax1.scatter(frame_numbers, psnr_values_sorted, marker='x', color='r', label="PSNR")
    best_fit_psnr = np.poly1d(np.polyfit(frame_numbers, psnr_values_sorted, best_fit_line_order))
    ax1.plot(frame_numbers, best_fit_psnr(frame_numbers), linestyle='dotted', linewidth=1, color='r')
    ax1.tick_params(axis='y', labelcolor="r")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add secondary axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel("bits", color="b", fontsize=12)
    ax2.scatter(frame_numbers, frame_bits_sorted, marker='o', color='b', label="bits")

    best_fit_mae = np.poly1d(np.polyfit(frame_numbers, frame_bits_sorted, best_fit_line_order))
    ax2.plot(frame_numbers, best_fit_mae(frame_numbers), linestyle='dotted', linewidth=1, color='b')
    ax2.tick_params(axis='y', labelcolor="b")

    axis_formatter = matplotlib.ticker.FuncFormatter(data_formatter_func)
    ax2.yaxis.set_major_formatter(axis_formatter)

    # Title and Layout
    plt.title(f"RD Curve: PSNR & Frame Size vs frame-index\nI-Period={params.encoder_config.I_Period}", fontsize=12)
    fig.tight_layout()
    plt.savefig(file_io.get_file_name("psnr_a.png"))

def plot_psnr_vs_frame_bits_b(file_io, frame_bytes, psnr_values):

    frame_bits = np.array(frame_bytes) * 8
    psnr_values = np.array(psnr_values)
    sorted_indices = np.argsort(frame_bits)

    frame_bits_sorted = frame_bits[sorted_indices]
    psnr_values_sorted = psnr_values[sorted_indices]
    best_fit_line_order = 3

    # Create plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot PSNR
    ax1.set_xlabel("Encoded Frame Size (bits)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", color="r", fontsize=12)
    ax1.scatter(frame_bits_sorted, psnr_values_sorted, marker='x', color='r', label="PSNR")
    best_fit_psnr = np.poly1d(np.polyfit(frame_bits_sorted, psnr_values_sorted, best_fit_line_order))
    ax1.plot(frame_bits_sorted, best_fit_psnr(frame_bits_sorted), linestyle='dotted', linewidth=1, color='r',
             label="PSNR Fit")
    ax1.tick_params(axis='y', labelcolor="r")
    ax1.grid(True, linestyle='--', alpha=0.6)


    # Title and Layout
    plt.title("RD Curve: PSNR  vs Frame Size", fontsize=14)
    fig.tight_layout()
    plt.savefig(file_io.get_file_name("psnr_b.png"))

data_formatter_func = lambda x, pos: '%1dMb' % (x * 1e-6) if x >= 1e6 else '%1dKb' % (x * 1e-3) if x >= 1e3 else '%1d' % x


class RDPointSeries:
    def __init__(self, series_name):
        self.series_name = series_name
        self.rd_points = []  # List to hold (total_bits, avg_psnr)

    def add_point(self, total_bits, avg_psnr):
        self.rd_points.append((total_bits, avg_psnr))

    def get_points(self):
        return self.rd_points

def load_frame_metrics(file_path):
    """Load FrameMetrics from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def calculate_rd_points_and_times(files, series_name):
    """Calculate RD points and encoding times for a series of files."""
    rd_points = RDPointSeries(series_name)
    encoding_times = []
    for file in files:
        df = load_frame_metrics(file)
        total_bits = df['file_bits'].sum()
        avg_psnr = df['PSNR'].mean()
        encoding_time = df.iloc[-1]['elapsed_time']
        rd_points.add_point(total_bits, avg_psnr)
        encoding_times.append(encoding_time)
    return rd_points, encoding_times


def plot_rd_curves( series_collection):
    """Plot multiple RD curves from a collection of RD point series."""
    plt.figure(figsize=(8, 5))
    for series in series_collection:
        rd_points = series.get_points()
        total_bits, avg_psnr = zip(*rd_points)
        plt.plot(total_bits, avg_psnr, marker='x', label=series.series_name)

    plt.title("Rate-Distortion Curves")
    plt.xlabel("Total Bitstream Size (bits)")
    plt.ylabel("Average PSNR (dB)")
    plt.legend()
    plt.grid(True)
    # plt.show()

    plt.savefig(f"../data/assign3_dels/ex2_rd.png")
    plt.close('all')

def tabulate_and_export_encoding_times(series_collection, encoding_times_collection):
    """Tabulate encoding times for the experiments and export to CSV."""
    # Prepare data for the table and CSV
    table = PrettyTable()
    table.field_names = ["Series", "Bits (bits)", "PSNR (dB)", "Encoding Time (s)"]

    csv_data = [["Series", "Bits (bits)", "PSNR (dB)", "Encoding Time (s)"]]

    for series, times in zip(series_collection, encoding_times_collection):
        for rd_point, time in zip(series.get_points(), times):
            bits, psnr = rd_point
            row = [series.series_name, bits, f"{psnr:.2f}", f"{time:.2f}"]
            table.add_row(row)
            csv_data.append(row)

    # Print the table
    print(table)

    # Write to CSV
    output_csv_path = f"../data/assign3_dels/timings.csv"
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print(f"Encoding time table saved to {output_csv_path}")


def plot_per_frame_psnr(metric_files):

    fig, ax1 = plt.subplots(figsize=(8,5))

    for idx, file in enumerate(metric_files):
        df = pd.read_csv(file)
        frames = df['idx']
        psnr_values = df['PSNR']
        plt.plot(frames, psnr_values, marker='o', label=f"RCflag {idx+1}")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Configure plot
    plt.title(f"Per-Frame PSNR for Bitrate = 2.4 Mbps")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../data/assign3_dels/per-frame-psnr.png")
    plt.close('all')

