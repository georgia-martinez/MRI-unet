"""
Script for calculating the average performance of all models
e.g. average performance of AC1 tested on AC2, AC1 tested on AC3, etc.

IMPORTANT: Run metrics_2D.py first
"""

import pandas as pd
import numpy as np
import csv

# exp_num = 11
# metric_file = "metric_results_2D" 

# models = ["AC1", "AC2", "AC3", "BC1", "BC2", "BC3", "WC1", "WC2", "WC3"]

# path = f"/home/gcm49/CF_Project/analyze_results/{metric_file}/experiment{exp_num}"

def performance_by_model(path):
    models = ["AC1", "AC2", "AC3", "BC1", "BC2", "BC3", "WC1", "WC2", "WC3"]

    with open(f"{path}/overall_results.csv", "w") as csv_file:
        writer = csv.writer(csv_file)

        header = ["Test Set", "Precision", "Recall", "Accuracy", "IoU", "F1"]
        writer.writerow(header)

        for model in models:
            first_letter = model[0]
            test_sets = [x for x in models if x[0] == first_letter and x[-1] != model[-1]]
            test_sets.append("external")

            for test in test_sets:
                if "external" not in test:
                    test_csv = f"{model}_testedOn_{test}.csv"
                else:
                    test_csv = f"{model}_external.csv"

                df = pd.read_csv(f"{path}/{test_csv}")
                
                test_name = f"{model} tested on {test}"
                row = [test_name]

                print(test_name)

                for col in df.columns:
                    if col == "Patient":
                        continue

                    metric = df[col].to_numpy()

                    avg = np.around(np.average(metric), 3)
                    std = np.around(np.std(metric), 3)

                    row.append(f"{avg} ± {std}")
                    print(f"{col}: {avg} ± {std}")
                print()

                writer.writerow(row)