"""
block_sizes = [8, 16]
search_range = 2
qp_values = {8: [0, 3, 6, 9], 16: [1, 4, 7, 10]}
I_Periods = [1, 4, 10]
num_frames = 10
input_file = 'data/foreman_cif.yuv'  
width = 352
height = 288

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

def run_experiment(block_size, qp, I_Period):
    encoder_config = EncoderConfig(
        block_size=block_size,
        search_range=search_range,
        quantization_factor=qp,
        I_Period=I_Period
    )

    input_params = InputParameters(
        y_only_file=input_file,
        width=width,
        height=height,
        encoder_config=encoder_config,
        frames_to_process=num_frames
    )

    start_time = time.time()
    ex4.main(input_params)  
    elapsed_time = time.time() - start_time
    psnr_values, bit_counts = collect_metrics()
    return elapsed_time, psnr_values, bit_counts

def collect_metrics():
    psnr_values = []
    bit_counts = []

    with open('results.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            frame_number, mae, psnr, bit_count = row
            psnr_values.append(float(psnr))
            bit_counts.append(int(bit_count))
    return psnr_values, bit_counts

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Block Size', 'QP', 'I_Period', 'PSNR', 'Bit Count', 'Elapsed Time (s)'])
        for result in results:
            for psnr, bit_count in zip(result['psnr_values'], result['bit_counts']):
                writer.writerow([result['block_size'], result['qp'], result['I_Period'], psnr, bit_count, result['elapsed_time']])

results = []
for block_size in block_sizes:
    for I_Period in I_Periods:
        for qp in qp_values[block_size]:
            print(f"Running experiment: Block Size={block_size}, QP={qp}, I_Period={I_Period}")
            elapsed_time, psnr_values, bit_counts = run_experiment(block_size, qp, I_Period)

            results.append({
                'block_size': block_size,
                'qp': qp,
                'I_Period': I_Period,
                'psnr_values': psnr_values,
                'bit_counts': bit_counts,
                'elapsed_time': elapsed_time
            })

save_results_to_csv(results, os.path.join(output_dir, 'rd_experiment_results.csv'))


def plot_rd_curve(results, block_size, I_Period):
    plt.figure(figsize=(10, 6))
    for result in results:
        if result['block_size'] == block_size and result['I_Period'] == I_Period:
            total_bit_count = sum(result['bit_counts'])
            avg_psnr = np.mean(result['psnr_values'])
            plt.plot(total_bit_count, avg_psnr, 'o-', label=f'QP={result["qp"]}')

    plt.xlabel("Total Bit Count (bits)")
    plt.ylabel("PSNR (dB)")
    plt.title(f'R-D Curve for Block Size={block_size} and I_Period={I_Period}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"rd_curve_block_{block_size}_I_Period_{I_Period}.png")
    plt.show()

def plot_bitcount_vs_frame(results, block_size, qp):
    plt.figure(figsize=(10, 6))
    for result in results:
        if result['block_size'] == block_size and result['qp'] == qp:
            plt.plot(range(1, num_frames+1), result['bit_counts'], label=f'I_Period={result["I_Period"]}')

    plt.xlabel("Frame Index")
    plt.ylabel("Bit Count (bits)")
    plt.title(f'Bit Count vs Frame Index for Block Size={block_size} and QP={qp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"bitcount_vs_frame_block_{block_size}_QP_{qp}.png")
    plt.show()

for block_size in block_sizes:
    for I_Period in I_Periods:
        plot_rd_curve(results, block_size, I_Period)

plot_bitcount_vs_frame(results, 8, 3)  # Block size 8, QP=3
plot_bitcount_vs_frame(results, 16, 4)  # Block size 16, QP=4






"""

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import ex4
from encoder.params import EncoderConfig
from input_parameters import InputParameters

block_sizes = [8, 16]
search_range = 2
qp_values = {8: [0, 3, 6, 9], 16: [1, 4, 7, 10]}
I_Periods = [1, 4, 10]
num_frames = 10
input_file = '../data/foreman_cif.yuv'
width = 352
height = 288

output_dir = '../results'
os.makedirs(output_dir, exist_ok=True)


def run_experiment(block_size, qp, I_Period):
    # Directory naming convention: blocksize_searchrange_qp
    experiment_dir = f'data/foreman_cif/{block_size}_{search_range}_{qp}'
    metrics_csv_path = os.path.join(experiment_dir, 'metrics.csv')

    encoder_config = EncoderConfig(
        block_size=block_size,
        search_range=search_range,
        quantization_factor=qp,
        I_Period=I_Period
    )

    input_params = InputParameters(
        y_only_file=input_file,
        width=width,
        height=height,
        encoder_config=encoder_config,
        frames_to_process=num_frames
    )

    start_time = time.time()
    ex4.main(input_params)  # Assuming it runs the encoding and saves the results
    elapsed_time = time.time() - start_time

    # Collect metrics from the corresponding CSV file
    psnr_values, bit_counts, total_bit_count = collect_metrics(metrics_csv_path)
    return elapsed_time, psnr_values, bit_counts, total_bit_count


def collect_metrics(metrics_csv_path):
    psnr_values = []
    bit_counts = []
    total_bit_count = 0

    with open(metrics_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            frame_number, mae, psnr, bit_count, total_bits, qp, I_Period = row
            psnr_values.append(float(psnr))
            bit_counts.append(int(bit_count))
            total_bit_count = int(total_bits)  # Take total bit count
    return psnr_values, bit_counts, total_bit_count


def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Block Size', 'QP', 'I_Period', 'PSNR', 'Bit Count', 'Total Bit Count', 'Elapsed Time (s)'])
        for result in results:
            for psnr, bit_count in zip(result['psnr_values'], result['bit_counts']):
                writer.writerow(
                    [result['block_size'], result['qp'], result['I_Period'], psnr, bit_count, result['total_bit_count'],
                     result['elapsed_time']])


results = []
for block_size in block_sizes:
    for I_Period in I_Periods:
        for qp in qp_values[block_size]:
            print(f"Running experiment: Block Size={block_size}, QP={qp}, I_Period={I_Period}")
            elapsed_time, psnr_values, bit_counts, total_bit_count = run_experiment(block_size, qp, I_Period)

            results.append({
                'block_size': block_size,
                'qp': qp,
                'I_Period': I_Period,
                'psnr_values': psnr_values,
                'bit_counts': bit_counts,
                'total_bit_count': total_bit_count,
                'elapsed_time': elapsed_time
            })

save_results_to_csv(results, os.path.join(output_dir, 'rd_experiment_results.csv'))


def plot_rd_curve(results, block_size, I_Period):
    plt.figure(figsize=(10, 6))
    for result in results:
        if result['block_size'] == block_size and result['I_Period'] == I_Period:
            total_bit_count = result['total_bit_count']  # Use total bit count from the experiment
            avg_psnr = np.mean(result['psnr_values'])
            plt.plot(total_bit_count, avg_psnr, 'o-', label=f'QP={result["qp"]}')

    plt.xlabel("Total Bit Count (bits)")
    plt.ylabel("PSNR (dB)")
    plt.title(f'R-D Curve for Block Size={block_size} and I_Period={I_Period}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"rd_curve_block_{block_size}_I_Period_{I_Period}.png")
    plt.show()


def plot_bitcount_vs_frame(results, block_size, qp):
    plt.figure(figsize=(10, 6))
    for result in results:
        if result['block_size'] == block_size and result['qp'] == qp:
            plt.plot(range(1, num_frames + 1), result['bit_counts'], label=f'I_Period={result["I_Period"]}')

    plt.xlabel("Frame Index")
    plt.ylabel("Bit Count (bits)")
    plt.title(f'Bit Count vs Frame Index for Block Size={block_size} and QP={qp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"bitcount_vs_frame_block_{block_size}_QP_{qp}.png")
    plt.show()


for block_size in block_sizes:
    for I_Period in I_Periods:
        plot_rd_curve(results, block_size, I_Period)

plot_bitcount_vs_frame(results, 8, 3)  # Block size 8, QP=3
plot_bitcount_vs_frame(results, 16, 4)  # Block size 16, QP=4
