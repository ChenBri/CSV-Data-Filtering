import sys
import csv
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import os
import collections

COLUMNS_TO_MODIFY = {"Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Engagement"}


def read_csv(file_path: str) -> List[Dict[str, float]]:
    print(f"Reading file: {file_path}")

    with open(file_path, encoding="utf-8-sig") as f:
        for _ in range(24):
            next(f)
        reader = csv.DictReader(f)

        return list(reader)[2:-2]


def write_data_to_csv(output_dir: Path(), original_file_name: Path, data: Dict[str, List]):
    output_path = output_dir / original_file_name.name

    print(f"Writing output file: {output_path}")

    writable_data = [(header, *values) for header, values in data.items()]
    with open(output_path, "w+", newline="") as f:
        writer = csv.writer(f)

        for row in zip(*writable_data):
            writer.writerow(row)

    # filter_data(output_path)


# def filter_data(path):
#     df = pd.read_csv(path)
#     df = df.iloc[2:-2]
#     df.to_csv(path, index=False)


def modify_data(path, data: List[Dict[str, float]]) -> Optional[List[Dict[str, float]]]:
    print(f"Processing...")

    stats_num_of_zeros = 0
    stats_zeros_freq = []

    dataset_size = len(data)
    formatted_data = defaultdict(list)

    for row in data:
        for header, value in row.items():
            if not value:
                value = 0.0
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            formatted_data[header].append(value)

    for col, data in formatted_data.items():
        if col not in COLUMNS_TO_MODIFY:
            continue

        col_data = formatted_data[col]
        zeros_count = 0
        zeros_groups = []  # 1

        result = []
        grouped_data = [(k, len(list(v))) for k, v in groupby(col_data)]

        for i, grp_res in enumerate(grouped_data):
            value, grp_size = grp_res

            if value == 0:
                zeros_groups.append(grp_size)  # 1
                stats_zeros_freq = zeros_groups

                zeros_count += grp_size  # 2
                stats_num_of_zeros = zeros_count

                if grp_size <= 150:
                    if i == 0:
                        result.extend([grouped_data[i + 1][0]] * grp_size)
                    elif i == len(grouped_data) - 1:
                        result.extend([grouped_data[i - 1][0]] * grp_size)
                    else:
                        result.extend([grouped_data[i - 1][0]] * (grp_size // 2))
                        result.extend([grouped_data[i + 1][0]] * (grp_size - grp_size // 2))
                else:
                    result.extend([value] * grp_size)
            else:
                result.extend([value] * grp_size)

        formatted_data[col] = list(result)

    if zeros_count / dataset_size >= 0.1:
        print_stats(stats_num_of_zeros, dataset_size, stats_zeros_freq, False, path)
        return None
    else:
        print_stats(stats_num_of_zeros, dataset_size, stats_zeros_freq, True, path)
        return formatted_data


def print_stats(zeros_count, dataset_size, freq_array, data_validity: bool, file_path) -> bool:
    zeros_percentage = round(zeros_count / dataset_size * 100, 3)
    freq_array_sorted = sorted(freq_array)

    headers = ['File Name', 'Data Size:', 'Number of Zeros:', 'Number of Zeros (%):', 'Data Validity:',
               'Frequency Array:', 'Sorted Frequency Array:']
    data = [file_path.name, dataset_size, zeros_count, f"{zeros_percentage}%", data_validity, freq_array,
            freq_array_sorted]
    csv_data = zip(headers, data)

    data_file_name = file_path.name
    data_file_name = os.path.splitext(data_file_name)[0]

    output_dir = Path.cwd() / sys.argv[2]

    os.makedirs(output_dir / "Data", exist_ok=True)

    with open(output_dir / "Data" / f"{data_file_name}_NAinfo.csv", "w+") as f:
        writer = csv.writer(f, lineterminator='\n')

        # Data

        writer.writerow(['#DATA'])

        for row in zip(*csv_data):
            writer.writerow(row)
        writer.writerow([])

        # Frequency Table

        writer.writerow(['#FREQUENCY TABLE'])
        writer.writerow(['Length', 'Frequency'])

        frequency_list = {}

        for item in freq_array_sorted:
            if item in frequency_list:
                frequency_list[item] += 1
            else:
                frequency_list[item] = 1

        for frequency in frequency_list.items():
            writer.writerow(frequency)


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_directory_path> <output_directory_path>")
        return

    input_dir, output_dir = Path(sys.argv[1]), Path.cwd() / sys.argv[2]
    output_dir.mkdir(exist_ok=True, parents=True)

    bad_files = []

    for data_file in input_dir.glob("*.csv"):
        if not data_file.is_file():
            continue

        data = read_csv(data_file)
        modified_data = modify_data(data_file, data)

        if modified_data is None:
            # print(f"Too many zero values in file: {data_file.name}")
            bad_files.append(data_file.name)
        else:
            write_data_to_csv(output_dir, data_file, modified_data)

        if bad_files:
            with open(output_dir / "bad_files.txt", "w+") as f:
                f.write("\n".join(bad_files))


if __name__ == "__main__":
    main()
