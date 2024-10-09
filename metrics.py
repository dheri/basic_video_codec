from matplotlib import pyplot as plt


def plot_metrics(avg_mae, file_prefix, block_size, search_range):
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
    plt.savefig(f'{file_prefix}_mae.png')

    plt.close()
    # Write the MAE values to a CSV file
    csv_file = f'{file_prefix}_mae.csv'
    with open(csv_file, 'wt') as f:
        for frame, mae in zip(frame_numbers, avg_mae_values):
            f.write(f'{frame}, {mae}\n')  # Write frame number and corresponding MAE value
