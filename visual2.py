import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to process and collect data from CSV files in a directory
def collect_data_from_directory(directory):
    # Get a list of CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Create a list to store data from all CSV files
    all_data = []

    # Read and store data from each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(directory, csv_file)
        data = pd.read_csv(csv_path, header=None, names=['x', 'timestamp', 'y'])
        all_data.append(data['y'])

    # Concatenate data from all CSV files and calculate mean and standard deviation
    concatenated_data = pd.concat(all_data, axis=1)
    mean = concatenated_data.mean(axis=1)
    std = concatenated_data.std(axis=1)
    if "modqn" in directory or "mosac" in directory:
        return data['x'][1:49], mean[1:49], std[1:49]
    elif "pql" in directory:
        return data['x'][:48], mean[:48], std[:48]
    else:
        return data['x'][:50], mean[:50], std[:50]

# Main function to traverse through directories, collect data, and plot
# Main function to traverse through directories, collect data, and plot
def main(root_directory):
    x_vals = []
    mean_vals = []
    std_vals = []
    directory_names = []

    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            x, mean, std = collect_data_from_directory(subdir_path)
            x_vals.append(x)
            mean_vals.append(mean)
            std_vals.append(std)
            directory_names.append(subdir)  # Store directory name

    # Plot mean with shaded region for upper and lower bounds
    plt.figure(figsize=(10, 6))
    for i in range(len(x_vals)):
        plt.plot(x_vals[i], mean_vals[i], label=directory_names[i])  # Use directory name as label
        plt.fill_between(x_vals[i], mean_vals[i] - std_vals[i], mean_vals[i] + std_vals[i], alpha=0.3)

    plt.xlabel('step')
    plt.ylabel('value')
    plt.title('Hypervolume')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/Hypervolume.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    root_dir = "/Users/amineelblidi/Documents/Bachlor vorbereitung/code/TEST/data/Hypervolume"
    main(root_dir)
