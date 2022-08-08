from unicodedata import name
import numpy as np
import tensorflow as tf
import h5py
import FCN_metrics
import unet
import json

from keras.layers import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.optimizers import Adam
from time import time

# Important hyperparameters
image_height = 128
image_width = 128
num_channels = 1
epochs = 50
batch_size = 16
learning_rate = 3e-3

model_name = "average_1"
model_version = "v4"

full_model_name = model_name + model_version

model_out_path = f"/data/gcm49/experiment3/models/{full_model_name}.h5" 

# Paths to the trainining and validation hdf5 files
HDF5Path_train = f"/data/gcm49/experiment3/hdf5_files/{model_name}_train.h5"
HDF5Path_val = f"/data/gcm49/experiment3/hdf5_files/{model_name}_val.h5"

# Build the network
input_img = Input((image_width, image_height, num_channels), name="img")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

optimizer = Adam(learning_rate=learning_rate)

with strategy.scope():

    model = unet.get_unet(input_img, num_channels)
    model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])
    # model.summary()

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

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Turn off AutoShard
options =  tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

train_data = train_data.with_options(options)
val_data = val_data.with_options(options)

# Set callbacks

json_logs = []

json_file = open(f"{model_name}{model_version}_loss_log.json", mode="wt", buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, 
    logs: json_logs.append({"epoch": epoch, "loss": logs["loss"], "val loss": logs["val_loss"]}),
)
        
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, verbose=1)
early_stop = EarlyStopping(monitor="loss", patience=4, mode="auto", baseline=None)

callbacks = [early_stop, json_logging_callback]

# Run U-Net
print("Running the U-Net")

model.fit(train_data, validation_data=val_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

print(f"Saving the model to: {model_out_path}")
model.save(f"{model_out_path}")

# Save JSON file with training info
data = {}
data["name"] = full_model_name
data["path"] = model_out_path
data["image height"] = image_height
data["image width"] = image_width
data["num channels"] = num_channels
data["epochs"] = epochs
data["batch size"] = batch_size
data["learning rate"] = learning_rate
data["logs"] = json_logs

with open(f"json_logs/{full_model_name}.json", "w") as json_file:
    json.dump(data, fp=json_file, indent=4)