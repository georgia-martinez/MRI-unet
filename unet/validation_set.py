import h5py
import re
import random
import numpy as np

def train_val_split(raw_train, label_train, names, PERCENTAGE=10):
    """
    Given X_train and y_train, creates X_val and y_val from 10% of the given training data. The selected 10% is removed from X_train and y_train. 

    @return X_train, y_train, X_val, y_val
    """

    random.seed(100)

    # Get a list of all patients (list: UH_RectalCA_167, CCF_RectalCA_033, etc...)
    PATIENT_ID_REGEX = "[A-Z]{2}[A-Z]?_RectalCA_[0-9]{3}"
    all_patients = []

    for name in names:
        patient_id = re.search(PATIENT_ID_REGEX, name).group(0)
    
        if patient_id not in all_patients:
            all_patients.append(patient_id)

    # Select a percentange of the patients for the validation set
    k = len(all_patients) * PERCENTAGE // 100
    val_indicies = random.sample(range(len(all_patients)), k)
    val_names = [all_patients[i] for i in val_indicies]

    # print(val_names)

    # Grab the raw and label slices for the validation set
    raw_val = []
    label_val = []

    assert raw_train.shape == label_train.shape
    original_shape = raw_train.shape

    idx_to_remove = []

    for i, name in enumerate(names):
        patient_id = re.search(PATIENT_ID_REGEX, name).group(0)

        if patient_id in val_names:
            raw_val.append(raw_train[i, :, :])
            label_val.append(label_train[i, :, :])

            idx_to_remove.append(i)

    raw_train = np.delete(raw_train, idx_to_remove, axis=0)
    label_train = np.delete(label_train, idx_to_remove, axis=0)

    raw_val = np.array(raw_val)
    label_val = np.array(label_val)

    assert raw_train.shape == label_train.shape
    new_shape = raw_train.shape

    assert raw_val.shape == label_val.shape
    assert np.array_equal(np.subtract(np.array(original_shape), np.array(new_shape)), np.array([raw_val.shape[0], 0, 0]))

    return raw_train, label_train, raw_val, label_val

if __name__ == "__main__":
    HDF5Path_train = f"/data/gcm49/experiment7/hdf5_files/AC_1.h5"

    with h5py.File(HDF5Path_train, "r") as f:
        X_train = f["raw"][()]
        y_train = f["labels"][()]
        names = f["slice_names"][()]
        names = [name.decode("utf-8") for name in names]

    train_val_split(X_train, y_train, names)