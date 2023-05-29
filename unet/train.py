import yaml
import os
import numpy as np
import tensorflow as tf
import h5py
import json
import matplotlib.pyplot as plt

import FCN_metrics
import unet

from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import Adam

from validation_set import train_val_split

def train_unet(
        image_height, 
        image_width, 
        num_channels,
        max_epochs,
        batch_size, 
        learning_rate, 
        data_path, 
        model_out_path, 
        json_path=None):

    # Build the network
    input_img = Input((image_width, image_height, num_channels), name="img")

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    optimizer = Adam(learning_rate=learning_rate)

    model = unet.get_unet(input_img, num_channels)
    model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])

    print("Built the network \n")

    # Set callbacks
    json_logs = []

    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, 
        logs: json_logs.append({"epoch": epoch, "loss": logs["loss"], "val loss": logs["val_loss"]}),
    )
            
    early_stop = EarlyStopping(monitor="loss", patience=4, mode="auto", min_delta=0.01, baseline=None)

    callbacks = [early_stop, json_logging_callback]

    # Get the training and validation datasets
    with h5py.File(data_path, "r") as f:
        raw = f["raw"][()]
        labels = f["labels"][()]
        names = f["slice_names"][()]
        names = [name.decode("utf-8") for name in names]

    print("Creating the training, validation split...")
    X_train, y_train, X_val, y_val = train_val_split(raw, labels, names)

    print("Finished loading the training set and validation datasets\n")

    # Expanding dimensions from (D, H, W) to (D, H, W, C)
    X_train = np.expand_dims(X_train, 3)
    y_train = np.expand_dims(y_train, 3)

    X_val = np.expand_dims(X_val, 3)
    y_val = np.expand_dims(y_val, 3)

    assert X_train.shape == y_train.shape
    assert X_val.shape == y_val.shape

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}\n")

    # On the fly image augmentations

    data_gen_args = dict(rotation_range=30, vertical_flip=True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generater = image_datagen.flow(X_train, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=7)

    train_generator = zip(image_generater, mask_generator)
    print("Passed zipping!\n")

    # Save some sample augmentations
    for X_batch, y_batch in train_generator:
        # plot 9 images
        for i in range(0,15):
            plt.subplot(5, 5, 1 + i)
            plt.grid(False)
            aug_img = X_batch[i]
            aug_mask = y_batch[i]
            plt.imshow(aug_img.squeeze(), cmap="gray", interpolation="bilinear")
            plt.contour(aug_mask.squeeze(), colors="y", levels=[0.5])
        # save the plot
        plt.savefig("sample_augmentations", bbox_inches="tight")
        break

    # Run U-Net
    print("Running the U-Net")

    model.fit(train_generator, steps_per_epoch = len(X_train)//batch_size, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=callbacks)

    # Save the model
    file_name = data_path.split("/")[-1].split(".")[0]

    model_out_path = os.path.join(model_out_path, file_name + ".h5")

    print(f"Saving the model to: {model_out_path}")
    model.save(model_out_path)

    if json_path is None:
        return

    # Save JSON file with training info
    data = {}
    data["name"] = file_name
    data["path"] = model_out_path
    data["train path"] = data_path
    data["image height"] = image_height
    data["image width"] = image_width
    data["num channels"] = num_channels
    data["max_epochs"] = max_epochs
    data["batch size"] = batch_size
    data["learning rate"] = learning_rate
    data["logs"] = json_logs

    json_path = os.path.join(json_path, file_name + ".json")

    with open(json_path, "w") as json_file:
        json.dump(data, fp=json_file, indent=4)

    print(f"Saving info to {json_path}")


if __name__ == "__main__":

    with open("configs/train.yaml") as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    image_height = config["image_height"]
    image_width = config["image_width"]
    num_channels = config["num_channels"]
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    learning_rate = float(config["learning_rate"])

    # Model paths
    data_path = config["data_path"]
    model_out_path = config["model_out_path"]
    json_log_path = config["json_log_path"]

    model_paths = []

    # Train only one model
    if data_path[-3:] == ".h5": 
        model_paths.append(data_path)

    # Train all of the models in the folder
    else:
        for model_path in os.listdir(data_path):
            full_path = os.path.join(data_path, model_path)
            model_paths.append(full_path)

    for path in model_paths:
        train_unet(image_height, image_width, num_channels, max_epochs, batch_size, learning_rate, path, model_out_path, json_log_path)