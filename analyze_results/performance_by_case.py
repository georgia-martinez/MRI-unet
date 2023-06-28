"""
Script for calculating the average performance of all models organized by case 

An example of a case is AC external, AC internal, BC external, etc...
"""

import os
import pandas as pd
import csv
import numpy as np

class Metric:
    def __init__(self, name):
        self.name = name
        self.values = []

    def append_val(self, values):
        self.values.extend(values)

    def average(self):
        return np.sum(self.values) / len(self.values)

    def std(self):
        return np.std(self.values)

def get_overall_metrics(model, directory, is_external=True, debug=False):
    """
    @param model: e.g. AC, BC, or WC
    @param directory: path to the directory with the csv prediction files

    @return: dictionary of with the metric as a key
    """

    metrics_dict = dict.fromkeys(["Precision","Recall", "Accuracy", "IoU", "F1"])

    for file_name in os.listdir(directory):
        if file_name == "overall_results.csv" or model not in file_name:
            continue

        if is_external:
            if "external" not in file_name:
                continue
        else:
            if "external" in file_name:
                continue

        # Read the csv
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        # Iterate through each column
        for col in df.columns:
            if col in metrics_dict:
                if not metrics_dict[col]:
                    metrics_dict[col] = Metric(col)

                metrics_dict[col].append_val(df[col].to_numpy())

    return metrics_dict

def performance_by_case(metrics_path):
    exp_num = 11

    models = ["AC", "BC", "WC"]
    directory = f"analyze_results/metric_results_2D/experiment{exp_num}"

    with open(f"analyze_results/performance_by_case_exp{exp_num}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)

        header = ["Test Set", "Precision", "Recall", "Accuracy", "IoU", "F1"]
        writer.writerow(header)

        for model in models:
            for is_external in [True, False]:
                # metrics_dict = get_overall_metrics(model, directory, is_external)
                metrics_dict = get_overall_metrics(model, metrics_path, is_external)

                # Adding the index (e.g. AC external, WC internal)
                suffix = "external" if is_external else "internal"
                test_name = f"{model} {suffix}"

                row = [test_name]

                for key in metrics_dict:
                    metric = metrics_dict[key]

                    decimals = 2
                    average = np.around(metric.average(), decimals)
                    std = np.around(metric.std(), decimals)

                    row.append(f"{average} ± {std}")

                writer.writerow(row)