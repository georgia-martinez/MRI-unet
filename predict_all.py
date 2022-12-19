"""
Runs predictions on all 9 models 

python3 predict_all.py -e 9
"""

import os
import argparse
import tensorflow as tf

from predict import load_and_predict

# Setting up the parser
parser = argparse.ArgumentParser(description="Predict")
parser.add_argument("-e", "--experiment", type=str, metavar="", help="Experiment number to access the correct folder")
args = parser.parse_args()

exp_num = args.experiment

# Handle memory issues 
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Run predictions on all 9 models
models = ["AC_1", "AC_2", "AC_3", "BC_1", "BC_2", "BC_3", "WC_1", "WC_2", "WC_3"]

for model_name in models:

    # Make path to save predictions
    out_path = f"/data/gcm49/experiment{exp_num}/predictions/{model_name}" # path to save the model

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Creating new directory: {out_path}")

    # Get the names of the two testing sets 
    # e.g. if model is AC_1, will test on AC_2 and AC_3
    model_num = model_name.split("_")[1]
    first_letter = model_name[0]
    nums = ["1", "2", "3"]

    test_files = [f"{first_letter}C_{x}" for x in nums if x != model_num]
    print(test_files)

    load_and_predict(model_name, "external", exp_num, f"{out_path}/external_predictions.h5")
    load_and_predict(model_name, test_files[0], exp_num, f"{out_path}/{test_files[0]}_predictions.h5")
    load_and_predict(model_name, test_files[1], exp_num, f"{out_path}/{test_files[1]}_predictions.h5")