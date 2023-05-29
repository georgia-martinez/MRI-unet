# Cohort Finder with MRI

## Set up

```
cd CF-MRI-project
pip install -r requirements.txt
```

## Usage

### Creating the hdf5 files



### Training the models

A 2D U-Net is provided in the `unet` folder. To use it, set up the `configs/train.yaml` file. Below are descriptions of the different parameters:

- `data_path`: path to the hdf5 files with the training data. If a path to a directory is provided with n hdf5 files, then n different models will be trained for each hdf5 file provided. If a path to a single hdf5 file is provided, only one model will be trained.
- `model_out_path`: path to a directory to store the model. The default name for the model will be the same as the hdf5 file provided as the `data_path`.
- `json_log_path` (optional): path to store a JSON log with information on the training. The log includes all of the information in the `train.yaml` and the loss and validation loss for each epoch.

