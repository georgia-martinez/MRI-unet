import numpy as np
import h5py
import cv2
import os
import tensorflow as tf

from keras.models import load_model
from FCN_metrics import dice_coef, dice_coef_loss
from binarize import *

def load_and_predict(FCNModelPath, testingHDF5, predFileName):
    """Load FCN model and predict on a dataset
    
    Parameters
    ----------
    FCNModelPath : string - filepath to saved FCN model
    testingHDF5 : string - filepath to dataset
    predFileName: string - HDf5 file to which predictions will be saved
    
    Returns
    -------
    Returns nothing but saves predictions as HDF5 file
    """

    loaded_model = load_model(FCNModelPath, custom_objects={"dice_coef": dice_coef,"dice_coef_loss": dice_coef_loss})

    with h5py.File(testingHDF5, "r") as f:
        X_test = f["raw"][()]
        labels = f["labels"][()]
        test_image_fileanmes = f["raw_names"][()]

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

    if os.path.exists(predFileName):
        os.remove(predFileName)

    hd5f_file = h5py.File(predFileName, mode="w")
    hd5f_file.create_dataset("predictions", data=predictions)
    hd5f_file.create_dataset("binary_predictions", data=binary_images)
    hd5f_file.create_dataset("file_names", data=test_image_fileanmes)
    hd5f_file.close()

    print(f"Saved predictions to {predFileName}")

if __name__ == "__main__":

    model_name = "average_2"
    test_file = "average_2_test"

    PATH = "/data/gcm49/experiment3"

    model_path = f"{PATH}/models/{model_name}.h5"
    test_path = f"{PATH}/hdf5_files/{test_file}.h5"
    out_path = f"{PATH}/predictions/{model_name}/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Creating new directory: {out_path}")

    out_path += f"{test_file}_predictions.h5"

    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        load_and_predict(model_path, test_path, out_path)