"""
Script for cropping MRI images to the tumor region given a desired image size
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import random
import os
import math

max_width = -math.inf
max_height = -math.inf

def bounding_box(img_slice, mask_slice, final_height, final_width, filepath="", filename="", plot=False):
    """
    Creates a bounding box around the biggest component in a binary 2D image of specified size (final_height and final_width).
    For this function to work properly, the mask MUST be binary (i.e. only 1 non-zero value is within mask):

    @param img_slice: 2D array - actual image
    @param mask_slice: 2D array - binary slice containing the mask
    @param final_height: int - desired height of the cropped image
    @param final_width: int - desired width of the cropped image
    @param filepath: string - path to save the plots
    @param filename: string - filename of the plot to save
    @param plot: bool [optional] - default is false. If true, a plot is saved showing the original image/mask and the cropped image/mask. 
        
    @return bbox_image: 2D array - cropped image
    @return bbox_mask: 2D array - cropped mask 
    """

    # Assert the input image and mask are the same shape
    assert img_slice.shape == mask_slice.shape, f"ERROR: img_slice.shape={img_slice.shape}, mask_slice.shape={mask_slice.shape}"

    # Assert the desired cropping size is smaller than the actual image
    img_height, img_width = img_slice.shape # the original input image

    assert img_height >= final_height, "ERROR: cropping height is larger than the image height"
    assert img_width >= final_width, "ERROR: cropping width is larger than the image width"

    # Get indices of unique, non-zero values in mask 
    height_idx, width_idx = np.nonzero(mask_slice)
    height_idx = np.unique(height_idx)
    width_idx = np.unique(width_idx)

    # Find the the middle of the tumor
    middle = len(height_idx) // 2
    center_y = height_idx[middle]

    middle = len(width_idx) // 2
    center_x = width_idx[middle]

    # Find the start and end values for cropping
    x1 = center_x - final_width // 2
    y1 = center_y - final_height // 2

    x2 = x1 + final_width
    y2 = y1 + final_height

    # Make sure the crop won't go out of bounds
    if(x1 < 0):
        x2 += abs(x1)
        x1 = 0

    if(y1 < 0):
        y2 += abs(y1)
        y1 = 0 

    if(x2 > img_width):
        x1 -= x2 - img_width
        x2 = img_width

    if(y2 > img_height):
        y1 -= y2 - img_height
        y2 = img_height 

    # print(f"x1={x1}, x2={x2}, y1={y1}, y2={y2}")

    # Crop the image and the mask
    bbox_image = img_slice[y1:y2, x1:x2]
    bbox_mask = mask_slice[y1:y2, x1:x2]

    # Assert the crop does does not go out of bounds
    assert x1 >= 0, f"ERROR: x1 should be > 0, but x1={x1}" 
    assert y1 >= 0, f"ERROR: y1 should be > 0, but y1={y1}"
    assert x2 <= img_width, f"ERROR: x2 should be <= {img_width}, but x2={x2}" 
    assert y2 <= img_height, f"ERROR: y2 should be <= {img_height}, but y2={y2}"

    # Assert the image and mask are the same shape after cropping
    assert bbox_image.shape == bbox_mask.shape, f"ERROR: bbox_image.shape={bbox_image.shape}, bbox_mask.shape={bbox_mask.shape}"

    # Plot for debugging
    if(plot):
        fig, ax = plt.subplots(1, 4, figsize=(50, 20))

        # Show the original image with the patches
        ax[0].imshow(img_slice, cmap='gray')
        ax[0].add_patch(patches.Rectangle((min(width_idx),min(height_idx)),(max(width_idx)-min(width_idx)),(max(height_idx)-min(height_idx)), alpha=0.6, color="red"))
        ax[0].add_patch(patches.Rectangle((x1, y1),(bbox_image.shape[1]),(bbox_image.shape[0]), alpha=0.6, color="c"))

        # Show the mask with the patches
        ax[1].imshow(mask_slice, cmap='gray')
        ax[1].add_patch(patches.Rectangle((min(width_idx),min(height_idx)),(max(width_idx)-min(width_idx)),(max(height_idx)-min(height_idx)), alpha=0.6, color="red"))
        ax[1].add_patch(patches.Rectangle((x1, y1),(bbox_image.shape[1]),(bbox_image.shape[0]), alpha=0.6, color="c"))

        # Show the cropped image and mask
        ax[2].imshow(bbox_image, cmap='gray')
        ax[3].imshow(bbox_mask, cmap='gray')

        filename += "_bbox.png"

        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')

    # Assert the image size is correct
    assertion_msg = f"ERROR: The desired size was ({final_height}, {final_width}) but the actual size is {bbox_image.shape}"

    assert bbox_image.shape == (final_height, final_width), assertion_msg
    assert bbox_mask.shape == (final_height, final_width), assertion_msg

    return bbox_image, bbox_mask

if __name__ == "__main__":

    from make_hdf5 import all_data_paths, MRI_volume

    src_path = "/data/gcm49/MRQy_Data/"
    plot_dst_path = "hdf5_scripts/bbox_images/"

    image_paths, label_paths = all_data_paths(src_path)

    # Clear the plot_dst_path of any previous images
    if os.path.exists(plot_dst_path):
        for f in os.listdir(plot_dst_path):
            os.remove(os.path.join(plot_dst_path, f))

    # The desired width and height for cropping
    WIDTH = 128
    HEIGHT = 128

    # Loop through all images
    for patient_count, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        img_volume = MRI_volume(image_path)
        label_volume = MRI_volume(label_path)

        patient = re.search("[A-Z]{2}[A-Z]?_RectalCA_[0-9]{3}", image_path).group(0)

        depth, height, width = np.shape(img_volume)

        for slice_count, (image_slice, label_slice) in enumerate(zip(img_volume, label_volume)):
            label_slice = np.where(label_slice != 1, 0, label_slice) # grabbing only label 1 (tumor)   

            # Skip slice if current label has no tumor            
            if not np.any(label_slice):
                continue

            filename = f"{patient}({slice_count+1})"

            # Only generate a plot some of the time
            plot_flag = 1 == random.randint(1,500) # [1, 100] inclusive
            
            # Plot a specific patient and slice number for debugging
            # plot_flag = False
            # if patient == "UH_RectalCA_149" and slice_count == 106:

            plot_flag = True

            try:
                bbox_mask, bbox_img = bounding_box(image_slice, label_slice, HEIGHT, WIDTH, plot_dst_path, filename, plot=plot_flag)
            except Exception as e:
                print(e)
                print(f"Error is with {patient} slice {slice_count}")
                
            print(f"{patient} slice {slice_count+1} out of {depth}")

        break
        print(f"{patient_count+1} out of {len(image_paths)} patients cropped")