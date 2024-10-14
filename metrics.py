from matplotlib import pyplot as plt

from file_io import get_file_name


def plot_metrics(avg_mae, original_file, block_size, search_range, residual_approx_factor):
    graph_file_name = get_file_name(original_file,'mae.png', block_size, search_range, residual_approx_factor)
    csv_file_name = get_file_name(original_file,'mae.png', block_size, search_range, residual_approx_factor)
    avg_mae_values = avg_mae  # Extract avg_mae for each frame
    frame_numbers = range(1, len(avg_mae_values) + 1)  # Generate frame numbers

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
    # Write the MAE values to a CSV file
    csv_file_name
    with open(csv_file_name, 'wt') as f:
        for frame, mae in zip(frame_numbers, avg_mae_values):
            f.write(f'{frame}, {mae}\n')  # Write frame number and corresponding MAE value
