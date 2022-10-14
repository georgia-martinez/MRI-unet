import numpy as np
import h5py
import cv2
import os
import re
import argparse

from keras.models import load_model
from FCN_metrics import dice_coef, dice_coef_loss
from binarize import *

def get_paths(model_name, test_file, exp_num):

    PATH = f"/data/gcm49/experiment{exp_num}"

    model_path = f"{PATH}/models/{model_name}.h5"
    test_path = f"{PATH}/hdf5_files/{test_file}.h5"
    out_path = f"{PATH}/predictions/{model_name}/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Creating new directory: {out_path}")

    out_path += f"{test_file}_predictions.h5"

    return model_path, test_path, out_path

def internal_set_name(model_name):
    """
    Returns the name of the internal testing set based on the model name

    average_1 --> average_1_test
    worst_1v4 --> worst_1_test

    @returns: name of the internal testing set
    """

    internal_set = model_name

    if bool(re.search("[0-9]v[0-9]", model_name)):
        internal_set = internal_set[:-2]

    internal_set += "_test"

    return internal_set

def load_and_predict(model_name, test_set_name, exp_num):
    """
    Load FCN model and predicts on a dataset. Predictions are saved to an HDF5 file.
    
    @param model_path: string - filepath to saved FCN model
    @param test_path: string - filepath to dataset
    @param out_path: string - HDf5 file to which predictions will be saved    
    """

    model_path, test_path, out_path = get_paths(model_name, test_set_name, exp_num)

    loaded_model = load_model(model_path, custom_objects={"dice_coef": dice_coef,"dice_coef_loss": dice_coef_loss})

    with h5py.File(test_path, "r") as f:
        X_test = f["raw"][()]
        labels = f["labels"][()]
        test_image_fileanmes = f["slice_names"][()]

    predictions = loaded_model.predict(X_test, verbose=1)
    predictions = np.array(predictions)
    
    # Binarize the predictions
    # binary_images = binary_masks(labels, predictions)

    binary_images = []

    for i in range(predictions.shape[0]):
        image = predictions[i, :, :]
        ret, thresh = cv2.threshold(image, .3, 1, cv2.THRESH_BINARY)
        binary_images.append(thresh)

    print("Shape of predictions: " + str(predictions.shape))

    if os.path.exists(out_path):
        os.remove(out_path)

    hd5f_file = h5py.File(out_path, mode="w")
    hd5f_file.create_dataset("predictions", data=predictions)
    hd5f_file.create_dataset("binary_predictions", data=binary_images)
    hd5f_file.create_dataset("file_names", data=test_image_fileanmes)
    hd5f_file.close()

    print(f"Saved predictions to {out_path}")

# Setting up the parser
parser = argparse.ArgumentParser(description="Predict")

# parser.add_argument("-m", "--model", type=str, metavar="", help="Name of the model (e.g. average_1)")
parser.add_argument("-e", "--experiment", type=str, metavar="", help="Experiment number to access the correct folder")

args = parser.parse_args()

if __name__ == "__main__":

    models = ["AC_1", "AC_2", "AC_3", "BC_1", "BC_2", "BC_3", "WC_1", "WC_2", "WC_3"]

    for model_name in models:
        exp_num = args.experiment

        model_num = model_name.split("_")[1]
        first_letter = model_name[0]
        nums = ["1", "2", "3"]

        test_files = [f"{first_letter}C_{x}" for x in nums if x != model_num]
        print(test_files)

        load_and_predict(model_name, "external", exp_num)
        load_and_predict(model_name, test_files[0], exp_num)
        load_and_predict(model_name, test_files[1], exp_num)