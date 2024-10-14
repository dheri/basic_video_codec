import csv

from matplotlib import pyplot as plt

from file_io import FileIOHelper
from input_parameters import InputParameters


def plot_metrics(params: InputParameters):
    file_io = FileIOHelper(params)

    csv_file_name = file_io.get_metrics_csv_file_name()
    frame_numbers = []
    avg_mae_values = []
    psnr_values = []

    # Read the CSV file and extract Frame Index, Average MAE, and PSNR
    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            frame, mae, psnr = row
            frame_numbers.append(int(frame))  # Frame index as integer
            avg_mae_values.append(float(mae))  # MAE value as float
            psnr_values.append(float(psnr))  # PSNR value as float

    # Generate frame numbers based on the number of MAE values
    # Plotting the metrics
    plt.figure(figsize=(10, 6))

    # Plot Average MAE
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')

    # Plot PSNR
    plt.plot(frame_numbers, psnr_values, marker='x', linestyle='--', color='r', label='PSNR')

    # Adding title and labels
    plt.title(f'MAE and PSNR per Frame, i = {params.block_size}, r = {params.search_range}, n = {params.residual_approx_factor}')
    plt.xlabel('Frame Number')
    plt.ylabel('Metric Value')

    # Adding grid, legend, and layout
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot as a PNG file (optional)
    graph_file_name = file_io.get_metrics_png_file_name()  # You might want to rename this method for clarity
    plt.savefig(graph_file_name)

    # Close the plot to avoid display issues in some environments
    plt.close()
