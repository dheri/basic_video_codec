import csv

from matplotlib import pyplot as plt

from file_io import FileIOHelper


def plot_metrics(file_io: FileIOHelper, block_size, search_range, residual_approx_factor):

    graph_file_name = file_io.get_mae_png_file_name()
    csv_file_name = file_io.get_mae_csv_file_name()

    # Read avg_mae_values from the CSV file
    avg_mae_values = []
    with open(csv_file_name, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            frame, mae = row
            avg_mae_values.append(float(mae))  # Convert MAE value to float

    # Generate frame numbers based on the number of MAE values
    frame_numbers = range(1, len(avg_mae_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, avg_mae_values, marker='o', linestyle='-', color='b', label='Avg MAE')

    plt.title(f'MAE per Frame, i = {block_size}, r = {search_range}')
    plt.xlabel('Frame Number')
    plt.ylabel('Average MAE')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(graph_file_name)

    plt.close()
