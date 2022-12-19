import numpy as np
import h5py
import cv2
import os
import re
import argparse
import tensorflow as tf

from keras.models import load_model
from FCN_metrics import dice_coef, dice_coef_loss
from binarize import *

def get_paths(model_name, test_file, exp_num):

    PATH = f"/data/gcm49/experiment{exp_num}"

    model_path = f"{PATH}/models/{model_name}.h5"
    test_path = f"{PATH}/hdf5_files/{test_file}.h5"

    return model_path, test_path

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

def load_and_predict(model_name, test_set_name, exp_num, out_path):
    """
    Load FCN model and predicts on a dataset. Predictions are saved to an HDF5 file.
    
    @param model_path: string - filepath to saved FCN model
    @param test_path: string - filepath to dataset
    @param out_path: string - HDf5 file to which predictions will be saved    
    """

    model_path, test_path = get_paths(model_name, test_set_name, exp_num)

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
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)

    # Handle memory issues 
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    # Run predictions
    exp_num = args.experiment

    model_name = "BC_1"
    test_set = "external"
    pred_file_name = f"{test_set}v2_predictions.h5"

    # Make path to save predictions
    out_path = f"/data/gcm49/experiment{exp_num}/predictions/{model_name}/" # path to save the model

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Creating new directory: {out_path}")

    out_path += pred_file_name

    load_and_predict(model_name, test_set, exp_num, out_path)