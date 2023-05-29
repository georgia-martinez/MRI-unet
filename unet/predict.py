import yaml
import numpy as np
import h5py
import cv2
import os
import tensorflow as tf

from keras.models import load_model
from FCN_metrics import dice_coef, dice_coef_loss


def load_and_predict(model_path, test_path, out_path):
    """
    Load FCN model and predicts on a dataset. Predictions are saved to an HDF5 file.
    
    @param model_path: string - filepath to saved FCN model
    @param test_path: string - filepath to dataset
    @param out_path: string - HDf5 file to which predictions will be saved    
    """

    loaded_model = load_model(model_path, custom_objects={"dice_coef": dice_coef,"dice_coef_loss": dice_coef_loss})

    with h5py.File(test_path, "r") as f:
        X_test = f["raw"][()]
        test_image_fileanmes = f["slice_names"][()]

    predictions = loaded_model.predict(X_test, verbose=1)
    predictions = np.array(predictions)

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


if __name__ == "__main__":
    # Handle memory issues 
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    # Get config settings
    with open("configs/predict.yaml") as f:
            config = yaml.safe_load(f)

    model_path = config["model_path"]
    test_file_path = config["test_file_path"]
    predict_out_path = config["predict_out_path"]

    if not os.path.exists(predict_out_path):
        os.makedirs(predict_out_path)
        print(f"Creating new directory: {predict_out_path}")

    model_paths = []

    # Predict only one model
    if model_path[-3:] == ".h5": 
        model_paths.append(model_path)

    # Test on only one model
    single_test_file = False

    if test_file_path[-3:] == ".h5": 
        single_test_file = True

    # Predict all models
    else:
        for path in os.listdir(model_path):
            full_path = os.path.join(model_path, path)
            model_paths.append(full_path)

    # Running predictions
    for model_path in model_paths:
        model_name = path.split("/")[-1].replace(".h5", "")

        out_path = os.path.join(predict_out_path, model_name)

        if single_test_file:
            test_model_name = test_file_path.split("/")[-1].replace(".h5", "")
            
            load_and_predict(model_path, test_file_path, os.path.join(out_path, f"{test_model_name}_predictions.h5"))

        else:
            # Get the names of the two testing sets 
            # e.g. if model is AC1, will test on AC2 and AC3
            model_num = model_name[-1]
            first_letter = model_name[0]
            nums = ["1", "2", "3"]

            test_files = [f"{first_letter}C{x}" for x in nums if x != model_num]

            load_and_predict(model_path, 
                            os.path.join(test_file_path, "external.h5"), 
                            os.path.join(out_path, "external_predictions.h5"))
            
            load_and_predict(model_path, 
                            os.path.join(test_file_path, test_files[0]), 
                            os.path.join(out_path, f"{test_files[0]}_predictions.h5"))
            
            load_and_predict(model_path, 
                            os.path.join(test_file_path, test_files[1]), 
                            os.path.join(out_path, f"{test_files[1]}_predictions.h5"))
