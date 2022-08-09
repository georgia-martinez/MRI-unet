import numpy as np
import tensorflow as tf
import h5py
import FCN_metrics
import unet
import json

from keras.layers import Input
from keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import Adam

# Important hyperparameters
image_height = 128
image_width = 128
num_channels = 1
epochs = 50
batch_size = 16
learning_rate = 3e-3

model_name = "worst_3"
model_version = "" # (e.g. v2 or v3 or leave blank)

full_model_name = model_name + model_version # (e.g. average_1 or average_1v4)

model_out_path = f"/data/gcm49/experiment3/models/{full_model_name}.h5" 

# Paths to the trainining and validation hdf5 files
HDF5Path_train = f"/data/gcm49/experiment3/hdf5_files/{model_name}_train.h5"
HDF5Path_val = f"/data/gcm49/experiment3/hdf5_files/{model_name}_val.h5"

# Build the network
input_img = Input((image_width, image_height, num_channels), name="img")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

optimizer = Adam(learning_rate=learning_rate)

model = unet.get_unet(input_img, num_channels)
model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])

print("Built the network \n")

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

# Set callbacks

json_logs = []

json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, 
    logs: json_logs.append({"epoch": epoch, "loss": logs["loss"], "val loss": logs["val_loss"]}),
)
        
early_stop = EarlyStopping(monitor="loss", patience=4, mode="auto", min_delta=0.01, baseline=None)

callbacks = [early_stop, json_logging_callback]

# Run U-Net
print("Running the U-Net")

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=callbacks)

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

with open(f"json_logs/{full_model_name}.json", "w") as json_file:
    json.dump(data, fp=json_file, indent=4)