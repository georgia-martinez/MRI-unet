# Cohort Finder with MRI

## Set up

```
cd CF-MRI-project
pip install -r requirements.txt
```

## Usage

### Creating the hdf5 files

The data must be in an hdf5 file format to use it with the provided 2D U-Net. The CohortFinder output should be organized into 4 different csv file: average case, best case, worst case, and the external set. Example csv files are provided in the `example_data` folder. You will need to format your csv files to match the examples since the CohortFinder output will not match this by default.

To put your data in hdf5 files, set up the `configs/make_hdf5.yaml` file. Then, run `data_prep/make_hdf5.py`.

IMPORTANT: Your external csv file should have "external" in the file name. You may want to modify the code to make it work better for your specific external csv format.

Below are descriptions of the different YAML parameters:

- `data_path`: path to the mha files that contain the MRI with its corresponding labels (there is a good chance you will need to edit some of the code inside `data_prep/make_hdf5.py` since some of it is based on a specific naming scheme for files).
- `output_path`: path to a directory to store the hdf5 files.
- `csv_path`: path to the csv file(s). If a path to a directory is provided, hdf5 files will be made for all of the csv files. If a path to a single csv file is provided, only hdf5 files for that file will be made.
- `img_size`: size that the images should be cropped to. The size will be img_size x img_size. This is because all of the images must be the same dimensions to work with the hdf5 file format.

For potential `data_path` debugging: here is an example file structure with file names that work with the code:

```
MRQy_Data/
├─ CCF_Pre_Resampled/
│  ├─ RectalCA_001
│  │  ├─ CCF_RectalCA_001_pre_ax_label_raw_resampled.mha  
│  │  ├─ CCF_RectalCA_001_pre_ax_raw_resampled.mha
│  ├─ RectalCA_002 ...
│  ├─ RectalCA_003 ...
├─ UH_Pre_Resampled/
│  ├─ RectalCA_001 ...
│  ├─ RectalCA_002 ...
│  ├─ RectalCA_003 ...
```

### Training the models

A 2D U-Net is provided in the `unet` folder. To use it, set up the `configs/train.yaml` file. Then, run `unet/train.py`. 

Below are descriptions of the different YAML parameters:

- `data_path`: path to the hdf5 files with the training data. If a path to a directory is provided with n hdf5 files, then n different models will be trained for each hdf5 file provided. If a path to a single hdf5 file is provided, only one model will be trained.
- `model_out_path`: path to a directory to store the model as an hdf5 file. The default name for the model will be the same as the hdf5 file provided as the `data_path`.
- `json_log_path` (optional): path to store a JSON log with information on the training. The log includes all of the information in the `train.yaml` and the loss and validation loss for each epoch.

### Predicting with the models

Once the models are trained, you can run predictions. If a specific model and a specific test file is provided, the model will run predictions on the test file. If a folder to many models and a test folder is provided, it will run three fold cross validation (e.g. AC1 will predict on AC2, AC3 | AC2 will predict on AC1 and AC3, etc...). To use it, set up the `configs/predict.yaml` file. Then, run `unet/predict.py`. 

Below are descriptions of the different YAML parameters:

- `model_path`: path to the trained model(s) stored in an hdf5 file. Can point towards a specific hdf5 file or a folder full of models.
- `test_file_path`: path to the hdf5 files to test on. Can point towards a specific hdf5 file or a folder containing test files. If a folder path is provided, the code will perform three fold cross validation based on file names.
- `predict_out_path`: path to a folder to store the predictions as hdf5 files.

### Analyzing results

Once predictions have been run, you can have metrics computed to evaluate the model performance. The metrics computed for this project were precision, accuracy, recall, IoU, and F1. To use it, set up the `configs/metrics_2D.yaml` file. Then, run `analyze_results/metrics_2D.py`. (Note that this script needs all the predictions to be generated from the previous step to run). This file will also generate a CSV containing overall metrics by case and by model

Below are descriptions of the different YAML parameters:
- `gt_folder_path`: path to the folder containing the ground truth h5 files.
- `pred_folder_path`: path to the folder containing the prediction h5 files (`predict_out_path` from the previous step).
- `metrics_output_path`: path to a folder where the metrics will be stored as CSV.