import tensorflow as tf
from tensorflow.python.client import device_lib
import os

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(os.path.abspath(tf.__file__))
print(get_available_devices()) 

# tf.config.set_visible_devices("GPU")
# print(tf.config.list_physical_devices("GPU"))

# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))