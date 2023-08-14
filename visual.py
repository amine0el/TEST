import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to process and plot CSV files in a directory
def process_and_plot_directory(directory):
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

    # Plot mean with shaded region for upper and lower bounds
    plt.plot(data['x'], mean, label='Mean')
    plt.fill_between(data['x'], mean - std, mean + std, alpha=0.3)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('hypervolume')
    plt.legend()
    plt.show()

# Main function to traverse through directories
def main(root_directory):
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            process_and_plot_directory(subdir_path)

if __name__ == "__main__":
    root_dir = "/Users/amineelblidi/Documents/Bachlor vorbereitung/code/TEST/data/Hypervolume"
    main(root_dir)