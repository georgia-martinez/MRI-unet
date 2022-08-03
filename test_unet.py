import load_and_predict as lp

# Test the FCN on both holdout testing sets
testRoot = "/home/tgd15/Post-Treatment/Revised SPIE/Experiments_Revised/Datasets/Testing/"

HDF5pathslist = [testRoot+"ORW_Testing_Dataset_expert1.hdf5",testRoot+"ORW_Testing_Dataset_expert2.hdf5", testRoot+"ORW_Testing_Dataset_excluded_masks.hdf5",testRoot+"ORW_Testing_Dataset_VA.hdf5"]
out_file_names = ["ORW_Testing_Predictions_expert1.hdf5","ORW_Testing_Predictions_expert2.hdf5","ORW_Preds_Excluded_Masks.hdf5", "ORW_Preds_VA.hdf5"]

model_name = "AC1.h5"
model_path = f"models/{model_name}"

for ind, dataset in enumerate(HDF5pathslist):
    lp.load_and_predict(model_name, dataset, out_file_names[ind])
print("Finished predicting on all holdout testing datasets.")