import matplotlib.pyplot as plt
import pandas as pd


# Function to read CSV files and extract the second column
def read_and_extract(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[:, 2]  # Extracting the second column


# List of file paths for CSV files in different locations
file_paths = [
    "./data/foreman_cif/8_2_0/metrics.csv"

    # Add more paths as needed
]

# List to store data from each file
data_list = []

# Read each CSV file and store the second column data
for file_path in file_paths:
    data_list.append(read_and_extract(file_path))

# Plotting the data from each CSV file
plt.figure(figsize=(10, 6))

for i, data in enumerate(data_list):
    plt.plot(data, label=f'File {i + 1}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Foreman CIF - PSNR - Varying i')
plt.legend()
plt.savefig(f"t.png")

plt.show()
