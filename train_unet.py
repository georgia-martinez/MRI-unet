import numpy as np
import tensorflow as tf
import datetime
import h5py
import tensorflow as tf
import FCN_metrics
import unet

from keras.layers import Input
from tensorflow.keras.optimizers import Adam

# Important hyperparameters
image_height = 128
image_width = 128
num_channels = 1
epochs = 10
batch_size = 32
learning_rate = 3e-3
model_name = "average_1"

# Paths to the trainining and validation hdf5 files
HDF5Path_train = f"/data/gcm49/experiment3/hdf5_files/{model_name}_train.h5"
HDF5Path_val = f"/data/gcm49/experiment3/hdf5_files/{model_name}_val.h5"

model_out_path = "/data/gcm49/experiment3/models"

# Build the network
input_img = Input((image_width, image_height, num_channels), name="img")

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
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

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Run U-Net
print("Running the U-Net")

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

print("Saving the model")
model.save(f"{model_out_path}/{model_name}.h5")