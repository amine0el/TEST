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
        data = pd.read_csv(csv_path)
        all_data.append(data['y'])

    # Concatenate data from all CSV files and calculate mean and standard deviation
    concatenated_data = pd.concat(all_data, axis=1)
    mean = concatenated_data.mean(axis=1)
    std = concatenated_data.std(axis=1)

    return data['x'], mean, std

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

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Hypervolume')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    root_dir = "/Users/amineelblidi/Documents/Bachlor vorbereitung/code/TEST/data/Hypervolume"
    main(root_dir)
