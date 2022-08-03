import random
import numpy as np
import tensorflow as tf
import datetime
import matplotlib
import h5py
from sklearn import metrics
import tensorflow as tf
import FCN_metrics
import unet
import sys
import load_and_predict as lp
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use("ggplot")
from time import time

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LambdaCallback
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy

###################################
#Handle some memory issues
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
###################################

###################################
#Important hyperparameters

image_height = 256
image_width = 256
num_channels = 1
epochs = 50
batch_size = 16
learning_rate = 3e-3
model_name = "AC1.hdf5"
###################################


###################################
#Build the network
input_img = Input((image_width, image_height, num_channels), name="img")

# with tf.device("/cpu:0"):
#     model = unet.get_unet(input_img, num_channels)
# parallel_unet = multi_gpu_model(model, gpus = 2)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

optimizer = Adam(learning_rate=learning_rate)

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
    model = unet.get_unet(input_img, num_channels)
    model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])
    model.summary()

# model.compile(optimizer=optimizer, loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])
# model.summary()
# parallel_unet.compile(optimizer=Adam(), loss=FCN_metrics.dice_coef_loss, metrics=[FCN_metrics.dice_coef])

print("Built the network \n")
###################################

###################################

# Load in training dataset
print("Loading the training dataset")

HDF5Path_train = "/data/gcm49/experiment2/hdf5_files/average_1_train.h5"
with h5py.File(HDF5Path_train, "r") as f:
    X_train = f["raw"][()]
    y_train = f["labels"][()]

print("Loading the validation dataset")

# Load in validation dataset
HDF5Path_val = "/data/gcm49/experiment2/hdf5_files/average_1_val.h5"
with h5py.File(HDF5Path_val, "r") as f:
    X_val = f["raw"][()]
    y_val = f["labels"][()]

print("Finished loading the training and validation datasets\n")
###################################

###################################
#Define Callbacks

# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: sys.stdout.write(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open("loss_log.json", mode="wt", buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({"epoch": epoch, "loss": logs["loss"]}) + "\n"),
    on_train_end=lambda logs: json_log.close()
)
        
callbacks = [ReduceLROnPlateau(factor=0.1, patience=5, verbose=1), TensorBoard(log_dir="logs/{}".format(time()),histogram_freq=10,write_images=True), json_logging_callback]

###################################

###################################
# Make sure training data is correct
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].imshow(X_train[127, ..., 0], cmap="gray", interpolation="bilinear")
# ax[0].contour(y_train[127].squeeze(), colors="y", levels=[0.5])
# ax[0].grid(False)
# ax[0].set_title("Mask Overlaied on Image")
# ax[1].imshow(y_train[127].squeeze(), interpolation="bilinear", cmap="gray")
# ax[1].grid(False)
# ax[0].set_title("Binary Mask")
# fig.savefig("vis.png", bbox_inches="tight")

# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].imshow(X_train[87, ..., 0], cmap="gray", interpolation="bilinear")
# ax[0].contour(y_train[87].squeeze(), colors="y", levels=[0.5])
# ax[0].grid(False)
# ax[0].set_title("Mask Overlaied on Image")
# ax[1].imshow(y_train[87].squeeze(), interpolation="bilinear", cmap="gray")
# ax[1].grid(False)
# ax[0].set_title("Binary Mask")
# fig.savefig("vis2.png", bbox_inches="tight")
# print("Ensured training data is correct. \n")
###################################

###################################

data_gen_args = dict(rotation_range=30,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Expanding dimensions from (D, H, W) to (D, H, W, C)
X_train = np.expand_dims(X_train, 3)
y_train = np.expand_dims(y_train, 3)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

image_datagen.fit(X_train, augment=True, seed=7)
mask_datagen.fit(y_train, augment=True, seed=7)

image_generater = image_datagen.flow(X_train, batch_size=32,seed=7) # batch size was originally 20
mask_generator = mask_datagen.flow(y_train, batch_size=32,seed=7)

train_generator = zip(image_generater, mask_generator)
print("Passed zipping!\n")

# Save some sample augmentations
for X_batch, y_batch in train_generator:
    # plot 9 images
    for i in range(0,19):
        plt.subplot(5,5, 1 + i)
        plt.grid(False)
        aug_img = X_batch[i]
        aug_mask = y_batch[i]
        plt.imshow(aug_img.squeeze(), cmap="gray", interpolation="bilinear")
        plt.contour(aug_mask.squeeze(), colors="y", levels=[0.5])
    # save the plot
    plt.savefig("Image Augmentations", bbox_inches="tight")
    break
###################################

###################################
# Run Unet
print("Running the U-Net")

results = model.fit_generator(train_generator, steps_per_epoch = len(X_train)//batch_size, epochs=epochs, validation_data=(X_val, y_val), validation_steps=(len(X_val)*2)//batch_size,callbacks=callbacks) # steps per epoch for training and validation were original 20
#results = parallel_unet.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_val, y_val),callbacks=callbacks)

print("Saving the model")
model.save(f"models/{model_name}")
###################################

###################################
# Plot loss vs. epochs
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig("results.png", bbox_inches="tight")
###################################

###################################
print("Evaluating on validation dataset...")
val_eval = model.evaluate(X_val, y_val, verbose=1)
print("Validation loss: " + str(val_eval[0]))
print("Validation DSC: " + str(val_eval[1]) + "\n")
###################################

###################################
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


###################################
# Test the FCN on both holdout testing sets
# testRoot = "/home/tgd15/Post-Treatment/Revised SPIE/Experiments_Revised/Datasets/Testing/"
# HDF5pathslist = [testRoot+"ORW_Testing_Dataset_expert1.hdf5",testRoot+"ORW_Testing_Dataset_expert2.hdf5", testRoot+"ORW_Testing_Dataset_excluded_masks.hdf5",testRoot+"ORW_Testing_Dataset_VA.hdf5"]
# out_file_names = ["ORW_Testing_Predictions_expert1.hdf5","ORW_Testing_Predictions_expert2.hdf5","ORW_Preds_Excluded_Masks.hdf5", "ORW_Preds_VA.hdf5"]
# for ind, dataset in enumerate(HDF5pathslist):
#     lp.load_and_predict(model_name, dataset, out_file_names[ind])
# print("Finished predicting on all holdout testing datasets.")