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
epochs = 1
batch_size = 128
learning_rate = 3e-3
model_name = "AC1.hdf5"

# Paths to the trainining and validation hdf5 files
HDF5Path_train = "/data/gcm49/experiment3/hdf5_files/average_1_train.h5"
HDF5Path_val = "/data/gcm49/experiment3/hdf5_files/average_1_val.h5"

# Build the network
input_img = Input((image_width, image_height, num_channels), name="img")

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

optimizer = Adam(learning_rate=learning_rate)

with strategy.scope():

    model = unet.get_unet(input_img, num_channels)
    model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])
    model.summary()

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

results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

print("Saving the model")
model.save(f"models/{model_name}")

# Evaluate on the validation datasest
print("Evaluating on validation dataset...")
val_eval = model.evaluate(X_val, y_val, verbose=1)

print("Validation loss: " + str(val_eval[0]))
print("Validation DSC: " + str(val_eval[1]) + "\n")

# Predict on training and validation
print("Predicting on training dataset \n")

preds_train = model.predict(X_train, verbose=1)
preds_train_t = (preds_train > 0.3).astype(np.float16)

print("Finished predicting on training dataset \n")

# Predict validation
print("Predicting on validation dataset \n")

preds_val = model.predict(X_val, verbose=1)
preds_val_t = (preds_val > 0.3).astype(np.float16)

print("Finished predicting on validation dataset \n")

preds_train_np = np.array(preds_train)
hd5f_file = h5py.File(datetime.datetime.today().strftime("%Y-%m-%d")+" FCN_Train_Val_Predictions.hdf5", mode="w")
hd5f_file.create_dataset("Pred_Masks", data = preds_val)
hd5f_file.create_dataset("Pred_Masks_T", data = preds_train)
hd5f_file.close()

print("Finished predicting on training and validation data. Saved predictions to HDF5 file. \n")