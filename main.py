import csv
import sys
import pandas as pd
import os
import statistics
from glob import glob
from pathlib import Path

pd.set_option("display.precision", 10)

# Set dataframe
stat_df = pd.DataFrame(
    columns=[
        "GroupNum",
        "ParID",
        "Segment",
        "Valence_Mean_Abs",
        "Valence_Mean",
        "Valence_Sum",
        "Valence_Sum_Abs",
    ]
)



# Open the CSV file, remove all the unnecessary rows and save the file name
def read_csv(file_path):
    filename = os.path.basename(file_path)
    filename_list = file_path.replace(".", "_").split("_")

    with open(file_path, encoding="utf-8-sig") as f:
        for _ in range(24):
            next(f)
        reader = csv.DictReader(f)

        modify_data(list(reader)[2:-2], filename_list)


# Modify the Valence column according to the value of the column Joy
def modify_data(data, filename):
    items_list = []

    for index, row in enumerate(data):
        if row["Joy"] != "0":
            items_list.append(float(row["Valence"]))

    calculate_statistics(items_list, filename)


# Create statistics and add them to the dataframe
def calculate_statistics(data_list, filename):
    abs_items_list = [abs(x) for x in data_list]

    data_mean = statistics.mean(data_list)
    data_abs_mean = statistics.mean(abs_items_list)
    data_sum = sum(data_list)
    data_abs_sum = sum(abs_items_list)

    stat_df.loc[len(stat_df.index)] = [
        filename[2],
        filename[3],
        filename[4],
        data_abs_mean,
        data_mean,
        data_sum,
        data_abs_sum,
    ]


# Choose the folder and modify each file
def selectFolder(folder):
    files = glob(folder + "/*/*", recursive=True)

    for file in files:
        try:
            read_csv(file)
        except:
            # If 'Joy' is always empty
            print(f"Corrupted file: {file}")

    print(stat_df)


# Call the function from the terminal
# Example: python main.py "C:/Users/chenb/Desktop/Data Stats/DataSamples" "C:/Users/chenb/Desktop/Data Stats"
# python main.py <input_directory_path> <output_directory_path>
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_directory_path> <output_directory_path>")
        return

    input_dir = str(sys.argv[1])
    output_dir = str(sys.argv[2]) + "/Statistics.csv"

    selectFolder(input_dir)
    stat_df.to_csv(output_dir, index=False)


if __name__ == "__main__":
    main()
