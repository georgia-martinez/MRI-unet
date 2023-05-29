import yaml
import os
import re
import numpy as np
import pandas as pd
import h5py
import SimpleITK as sitk 
from bounding_box import bounding_box

def all_data_paths(path):
    """
    Given a path to an MRI dataset, returns a list of the images paths and the corresponding labels paths

    @param path: path to files
    @returns:
        list of image data paths
        list label data paths
    """

    image_paths = []
    label_paths = []

    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)

            if "ax" in file_name:
                if "label" in file_name:
                    label_paths.append(file_path)

                else:
                    image_paths.append(file_path)

    validate_data(image_paths, label_paths)

    return image_paths, label_paths

PATIENT_ID_REGEX = "[A-Z]{2}[A-Z]?_RectalCA_[0-9]{3}"

def validate_data(image_paths, label_paths):
    """
    Given a list of images and labels, makes sure that the two lists align to the correct patient

    @param images: list of images to check
    @param labels: list of labels to check
    @return: true if valid, otherwise raises an exception
    """

    if len(image_paths) != len(label_paths):
        raise Exception("Lists must be the same length")
    
    for i in range(len(image_paths)):
        image_name = image_paths[i].split("/")[-1]
        image_id = re.search(PATIENT_ID_REGEX, image_name).group(0)

        label_name = label_paths[i].split("/")[-1]
        label_id = re.search(PATIENT_ID_REGEX, label_name).group(0)

        if image_id != label_id:
            raise Exception()

    return True

def patient_paths(patients, image_paths, label_paths):
    """
    Given a list of patients and a list of image paths and with a list of corresponding label paths,
    returns a list of the files for the given patients 
    """

    patient_image_paths = []
    patient_label_paths = []

    for i in range(len(patients)):
        for j in range(len(image_paths)):
            if patients[i] in image_paths[j]:
                patient_image_paths.append(image_paths[j])
                patient_label_paths.append(label_paths[j])

                break

    return patient_image_paths, patient_label_paths 

def patients_from_csv(csv_path, col_name):
    """
    Gets the list of patients from the specified column of a given csv file

    @param file_path: path to the csv file
    @param col_name: column to get data from
    @return: list of patients
    """

    df = pd.read_csv(csv_path, keep_default_na=False)
    patients = df[col_name]

    patients = [i for i in patients if i] # removing blank strings
    
    return patients

def MRI_volume(path):
    """
    Returns a 3D numpy array of the MRI volume given a path to the mha file

    @param path to mha file
    @return 3D numpy array of the MRI volume
    """

    image = sitk.ReadImage(path, imageIO="MetaImageIO")
    image = sitk.GetArrayFromImage(image)

    depth, height, width = np.shape(image)

    slices = [image[i, :, :] for i in range(depth)]

    return slices

def patient_data(image_paths, label_paths, img_size):
    """
    Gets the image volume from each path and returns the list of slices as numpy arrays and the corresponding slice names
 
    @param image_paths: list of image paths
    @param label_paths: list of label paths
    @param img_size: images will be cropped to img_size x img_size
    @return: 
        list of images as numpy arrays
        list of image names
    """

    images = [] 
    labels = []
    slice_names = []

    N = img_size # Cropped images will be N x N
    count = 1

    assert len(image_paths) == len(label_paths), "image_paths is a different size than label_paths"
    total_paths = len(image_paths)

    for image_path, label_path in zip(image_paths, label_paths):
        patient = re.search("[A-Z]{2}[A-Z]?_RectalCA_[0-9]{3}", image_path).group(0)

        print(f"({count} out of {total_paths})", end=" ")
        print(patient)

        label_slices = MRI_volume(label_path)

        image_slices = MRI_volume(image_path)

        patient_images = []
        patient_labels = []
        patient_slice_names = []

        for i, (image_slice, label_slice) in enumerate(zip(image_slices, label_slices)):

            curr_label = np.where(label_slice != 1, 0, label_slice) # grabbing only label 1 (tumor)   
            assert ((curr_label==0) | (curr_label==1)).all()

            # Skip slice if current label has no tumor            
            if not np.any(curr_label):
                continue

            bbox_image, bbox_label = bounding_box(image_slice, curr_label, N, N)

            patient_images.append(bbox_image)
            patient_labels.append(bbox_label)
            patient_slice_names.append(f"{patient}({i+1})".encode("utf8"))

        count += 1

        # Skip patient if label volume has no tumor annotated
        if len(patient_labels) != 0:
            images.extend(patient_images)
            labels.extend(patient_labels)
            slice_names.extend(patient_slice_names)
        else:
            print(f"WARNING: Skipping {patient} because there is no tumor annotated")

    slice_names = np.array(slice_names)

    return images, labels, slice_names

def create_hdf5(patients, file_name, src_path, dst_path, img_size=128):
    """
    Creates an hdf5 file which stores images, labels, image names, and label names

    @param patients: list of patients
    @param file_name: name for the hdf5 file
    @param src_path: path to the data
    @param dst_path: path to store the hdf5 file
    """

    print(f"Starting {file_name}")
    
    image_paths, label_paths = all_data_paths(src_path)
    patient_img_paths, patient_label_paths = patient_paths(patients, image_paths, label_paths)

    image_data, label_data, slice_names = patient_data(patient_img_paths, patient_label_paths, img_size)

    hdf5_path = dst_path + file_name + ".h5"
    hd5f_file = h5py.File(hdf5_path, "w")
    
    print("Creating the hdf5 file...")

    hd5f_file.create_dataset("raw", data=image_data)
    hd5f_file.create_dataset("labels", data=label_data)
    hd5f_file.create_dataset("slice_names", data=slice_names)
    
    hd5f_file.close()   

    print(f"Finished with {file_name}\n")


if __name__ == "__main__":

    with open("configs/make_hdf5.yaml") as f:
        config = yaml.safe_load(f)

    data_path = config["data_path"]
    output_path = config["output_path"]
    csv_path = config["csv_path"]

    img_size = config["img_size"]

    csv_paths = []

    # Path leads to one csv file
    if data_path[-3:] == ".csv": 
        csv_paths.append(data_path)

    # Path leads to a folder of csv files
    else:
        for file_path in os.listdir(csv_path):
            full_path = os.path.join(csv_path, file_path)
            csv_paths.append(full_path)

    # Create the hdf5 files
    for path in csv_paths:

        # External set
        if "external" in path:

            # Patients with a 1 belong to the external dataset
            df = pd.read_csv(path, keep_default_na=False)
            patients = df.loc[df["external_dataset"] == "1", "folder_name"]
            patients = [i for i in patients]

            create_hdf5(patients, "external", data_path, output_path)        

        # Internal set
        else:
            columns = pd.read_csv(path, keep_default_na=False, nrows=1).columns.tolist() # e.g. AC1, AC2, AC3
            
            for col_name in columns:
                patients = patients_from_csv(path, col_name)

                create_hdf5(patients, col_name, data_path, output_path, img_size)