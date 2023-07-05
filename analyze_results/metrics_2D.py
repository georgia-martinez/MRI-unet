import h5py
import re
import numpy as np
import csv
import yaml
import os

from performance_by_model import performance_by_model
from performance_by_case import performance_by_case

def get_h5data(gt_path, pred_path):
    """
    """

    with h5py.File(gt_path, "r") as gt_file:
        gt_masks = gt_file["labels"][()]
        names = gt_file["slice_names"][()]
        names = [name.decode("utf-8") for name in names]

    with h5py.File(pred_path, "r") as pred_file:
        pred_masks = pred_file["binary_predictions"][()]

    return gt_masks, pred_masks, names

def divide(x, y):
    return x / y if y != 0 else 0

def generate_metrics(gt_masks, pred_masks, names, output_csv_path):
    """
    """

    with open(f"{output_csv_path}", "w") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["Patient", "Precision", "Recall", "Accuracy", "IoU", "F1"])

        for i, patient in enumerate(names):
            groundtruth = gt_masks[i, :, :]
            premask = pred_masks[i, :, :]

            seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
            true_pos = np.logical_and(premask, groundtruth).sum()
            true_neg = np.logical_and(seg_inv, gt_inv).sum()
            false_pos = np.logical_and(premask, gt_inv).sum()
            false_neg = np.logical_and(seg_inv, groundtruth).sum()

            precision = divide(true_pos, true_pos + false_pos)
            recall = divide(true_pos, true_pos + false_neg)
            accuracy = divide(true_pos + true_neg, true_pos + false_pos + true_neg + false_neg)
            IoU = divide(true_pos, true_pos + false_neg + false_pos)
            F1 = divide(2 * true_pos, 2 * true_pos + false_pos + false_neg)

            # Create patient slice name e.g. UH_RectalCA_167(80)
            patient_id = re.search("[A-Z]{2}[A-Z]?_RectalCA_[0-9]{3}", patient).group(0)
            slice_num = re.search("\([0-9]{1}[0-9]?[0-9]?\)", patient).group(0)
            patient_name = patient_id + slice_num

            writer.writerow([patient_name, precision, recall, accuracy, IoU, F1])

    print(f"Metrics saved to: {output_csv_path}")

if __name__ == "__main__":

    with open("configs/metrics_2D.yaml") as f:
        config = yaml.safe_load(f)

    gt_folder_path = config["gt_folder_path"]
    pred_folder_path = config["pred_folder_path"]
    metrics_out_path = config["metrics_out_path"]

    model_paths = []

    # Compute metrics for all models in the folder
    for path in os.listdir(pred_folder_path):
        full_path = os.path.join(pred_folder_path, path)
        model_paths.append(full_path)

    # # Compute metrics for only one model
    # if pred_folder_path[-3:] == ".h5": 
    #     model_paths.append(pred_folder_path)

    # # Compute metrics for all models in the folder
    # else:
    #     for path in os.listdir(pred_folder_path):
    #         full_path = os.path.join(pred_folder_path, path)
    #         model_paths.append(full_path)

    for model in model_paths:

        model_name = model.split("/")[-1]
        first_letter = model_name[0]
        model_num = model_name[-1]

        nums = ["1", "2", "3"]
        test_files = [f"{first_letter}C{x}" for x in nums if x != model_num]

        print(f"Generating 2D metrics for the {model} model...")

        # Test on the external set
        gt_path = os.path.join(gt_folder_path, "external.h5")
        pred_path = os.path.join(pred_folder_path, model, "external_predictions.h5")

        gt_masks, pred_masks, names = get_h5data(gt_path, pred_path)

        # letter, version = model.split("_") #TODO: I don't think it's split by _ anymore
        # csv_name = f"{letter[0].upper()}C{version}_external.csv"

        csv_name = f"{model_name}_external.csv"
        csv_path = os.path.join(metrics_out_path, csv_name)

        generate_metrics(gt_masks, pred_masks, names, csv_path)

        # Test on the other two internal testing sets

        for test in test_files:
            gt_path = os.path.join(gt_folder_path, f"{test}.h5")
            pred_path = os.path.join(pred_folder_path, model, f"{test}_predictions.h5")

            gt_masks, pred_masks, names = get_h5data(gt_path, pred_path)

            csv_name = f"{model_name}_testedOn_{test}.csv" # e.g. AC1_testedOn_AC2.csv
            csv_path = os.path.join(metrics_out_path, csv_name)

            generate_metrics(gt_masks, pred_masks, names, csv_path)

    # Calculate overall metrics
    performance_by_model(metrics_out_path)
    performance_by_case(metrics_out_path)