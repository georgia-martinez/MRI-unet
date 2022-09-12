import numpy as np
import tensorflow as tf
import h5py
import FCN_metrics
import unet
import json
import argparse
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import Adam

# Setting up the parser
parser = argparse.ArgumentParser(description="Train")

parser.add_argument("-m", "--model", type=str, metavar="", help="Name of the model (e.g. average_1)")
parser.add_argument("-v", "--version", type=str, metavar="", nargs="?", default="", help="Version number of model (e.g. v2 or v2)")
parser.add_argument("-e", "--experiment", type=str, metavar="", help="Experiment number to access the correct folder")
parser.add_argument("-f", "--hdf5", type=str, metavar="", help="Experiment number to access the hdf5 files")

args = parser.parse_args()

# Important hyperparameters
image_height = 128
image_width = 128
num_channels = 1
epochs = 50
batch_size = 16
learning_rate = 3e-3

model_name = args.model
model_version = args.version
exp_num = args.experiment
hdf5_num = args.hdf5

full_model_name = model_name + model_version # (e.g. average_1 or average_1v4)

model_out_path = f"/data/gcm49/experiment{exp_num}/models/{full_model_name}.h5" 

# Paths to the training and validation hdf5 files
HDF5Path_train = f"/data/gcm49/experiment{hdf5_num}/hdf5_files/{model_name}_train.h5"
HDF5Path_val = f"/data/gcm49/experiment{hdf5_num}/hdf5_files/{model_name}_val.h5"

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

# Load in training dataset
print("Loading the training dataset")

with h5py.File(HDF5Path_train, "r") as f:
    X_train = f["raw"][()]
    y_train = f["labels"][()]

print("Loading the validation dataset")

# Load in validation dataset
with h5py.File(HDF5Path_val, "r") as f:
    X_val = f["raw"][()]
    y_val = f["labels"][()]

print("Finished loading the training and validation datasets\n")

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

data_gen_args = dict(rotation_range=30,
                     vertical_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=7)
mask_datagen.fit(y_train, augment=True, seed=7)

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

model.fit(train_generator, validation_data=(X_val, y_val), epochs=epochs, callbacks=callbacks)

# model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=callbacks)

print(f"Saving the model to: {model_out_path}")
model.save(f"{model_out_path}")

# Save JSON file with training info
data = {}
data["name"] = full_model_name
data["path"] = model_out_path
data["train path"] = HDF5Path_train
data["val path"] = HDF5Path_val
data["image height"] = image_height
data["image width"] = image_width
data["num channels"] = num_channels
data["epochs"] = epochs
data["batch size"] = batch_size
data["learning rate"] = learning_rate
data["logs"] = json_logs

with open(f"json_logs/experiment{exp_num}/{full_model_name}.json", "w") as json_file:
    json.dump(data, fp=json_file, indent=4)