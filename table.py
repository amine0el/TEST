import os
import pandas as pd

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
    max_values = concatenated_data.loc[47, :].mean()  # Get max value for index 48
    std_values = concatenated_data.loc[47, :].std()  # Get std value for index 48
    return max_values, std_values

# Main function to traverse through directories, collect data, and create a table
def main(root_directory):
    max_values_dict = {}
    std_values_dict = {}

    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            max_value, std_value = collect_data_from_directory(subdir_path)
            max_values_dict[subdir] = max_value
            std_values_dict[subdir] = std_value

    # Create a DataFrame with the max hypervolume values and standard deviations
    df_max = pd.DataFrame.from_dict(max_values_dict, orient='index', columns=['Mean HV'])
    df_std = pd.DataFrame.from_dict(std_values_dict, orient='index', columns=['Std Deviation'])

    # Combine the DataFrames
    df_combined = pd.concat([df_max, df_std], axis=1)

    # Print the combined DataFrame
    print(df_combined)
    df_combined.to_csv('table/HV.csv')

if __name__ == "__main__":
    root_dir = "/Users/amineelblidi/Documents/Bachlor vorbereitung/code/TEST/data/Hypervolume"
    main(root_dir)
