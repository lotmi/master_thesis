"""
This script presents the following pre-processing pipeline for TotalSegmentator CT scans: 

1. Windowing: apply a bone window to the HU values in the CT scan 
2. Normalizing: min-max normalization to bound the pixel values between 0 and 1
3. Slicing: creating 2D axial slices form the CT scans and extracting the corresponding bone bounding boxes
4. Formatting: putting images and labels in representations that YOLO and RetinaNet can work with

Uncomment the steps you want to execute/re-run.

"""

# Import packages
import os
import shutil
import random
import json
import numpy as np
import pandas as pd
import nibabel as nib
import skimage as skim 

# Defining datasplit (global variable)
# BASED ON SCAN INDEX: datasplit = {"train":(1, 166), "val":(167, 202), "test":(203, 237)} 
# This gives set magnitudes of 166, 36 and 35 respectively
# BASED ON SCAN NR:
datasplit = {"train":(4, 896), "val":(897, 1012), "test":(1016, 1152)}



"""------------------------------------- Scaling (step 1 + 2) --------------------------------------"""
from preprocessing_modules.scaling import apply_window, apply_normalization

# Access data 
inputpath = '/home/mleeuwen/DATA/TSv3_Selection/Images'
outputpath = '/home/u366836/thesis/DATA/TotalSegmentator/Scaled_Scans'
all_images = sorted(os.listdir(inputpath))

# Loop over and process all subjects
print("Scaling all CT scans.", flush=True)
for img_file in all_images: 
    scan_nr = img_file[5:9]
    img = nib.load(inputpath+'/'+img_file)
    img_arr = img.get_fdata()
    windowed_img = apply_window(img_arr, W=1800, L=400) # to check: print(windowed_img.min(), windowed_img.max(), flush=True) # should be between -500 and 1300
    normalized_img = apply_normalization(windowed_img) # to check: print(normalized_img.min(), normalized_img.max(), flush=True) # should be between 0 and 1
    np.save(f"{outputpath}/{scan_nr}_scaled.npy", normalized_img)
print("Completed scaling of all CT scans.", flush=True)

    
    
"""------------------------------------- Slicing (step 3) -----------------------------------------"""
from preprocessing_modules.slicing import create_slices 

# Access data
image_inputpath = '/home/u366836/thesis/DATA/TotalSegmentator/Scaled_Scans'
label_inputpath = '/home/mleeuwen/DATA/TSv3_Selection/Labels'
slice_outputpath = '/home/u366836/thesis/DATA/TotalSegmentator/YOLO/images'
props_outputpath = '/home/u366836/thesis/DATA/TotalSegmentator/slice_properties.csv' 
all_labels = sorted(os.listdir(label_inputpath))

# Define desired slice properties
properties = ['label', 'area', 'centroid', 'bbox'] 
props_dict = {'scan_nr':[], 'view':[], 'slice_index':[], 'index':[], 'label':[], 'area': [], 'centroid-0': [], 'centroid-1': [], 'bbox-0': [], 'bbox-1': [], 'bbox-2': [], 'bbox-3': [], 'width':[], 'height':[]} 
views = ['top']

# Loop over and process all scans
print("Creating {} slices for all CT scans.".format("& ".join(views)), flush=True)
for lab_file in all_labels:
    scan_nr = lab_file[5:9]
    label_nifti = nib.load(label_inputpath+'/'+lab_file)
    label = skim.measure.label(label_nifti.get_fdata().astype(int))
    img = np.load(f"{image_inputpath}/{scan_nr}_scaled.npy")
    props_dict = create_slices(scan_nr, img, label, props_dict, datasplit, slice_outputpath, views, properties)
print("Completed slicing all CT scans.", flush=True)    

# Save properties
props_table = pd.DataFrame(props_dict)
props_table.to_csv(props_outputpath)
print(f"Saved nodule properties for all slices to {props_outputpath}.", flush=True)


    
"""------------------------------------- B-Box Conversion (step 4) ---------------------------------"""
from preprocessing_modules.data_formatting import to_yolo_format, to_pascal_format

# Access data
props_inputpath = '/home/u366836/thesis/DATA/TotalSegmentator/slice_properties.csv'  

# YOLO Conversion
yolo_outputpath = '/home/u366836/thesis/DATA/TotalSegmentator/YOLO/labels'
print("Converting bounding boxes to YOLO format.", flush=True)
to_yolo_format(yolo_outputpath, props_inputpath, datasplit)
print("Completed YOLO bounding box conversion.", flush=True)

# PASCAL Conversion
pascal_outputpath = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL"
print("Converting bounding boxes to PASCAL format.", flush=True)
to_pascal_format(pascal_outputpath, props_inputpath, datasplit)
print("Completed PASCAL bounding box conversion.", flush=True)



"""----------------------------- Copying Images (not generalizeable) -------------------------------"""
# # Change paths to what you want
# shutil.copytree("/home/u366836/thesis/DATA/TotalSegmentator/YOLO/images/val", "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL/val/images", dirs_exist_ok=True)


"""------------------------ Sampling Images (for hyperparam. optimization) -------------------------"""

yolo_sample_output_path = "/home/u366836/thesis/DATA/TotalSegmentator/YOLO_sample/"
pascal_sample_output_path = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL_sample"
pascal_train_annotation_file = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL/train/annotations.json"
train_image_path = "/home/u366836/thesis/DATA/TotalSegmentator/YOLO/images/train"

train_images = all_images = sorted(os.listdir(train_image_path))
sample_ratio = 0.10 # 10% of 70438 images -> sample of 7044 images
sample_size = int(sample_ratio * len(train_images))

print("Creating training sample for hyperparameter optimization.", flush=True)
random.seed(42)
sample = random.sample(train_images, sample_size)
for image in sample:
    # Create YOLO sample
    file = f'{train_image_path}/{image}'
    label = f'/home/u366836/thesis/DATA/TotalSegmentator/YOLO/labels/train/{image.strip('.png')}.txt'
    shutil.copyfile(file, f'{yolo_sample_output_path}/images/train/{image}')
    shutil.copyfile(label, f'{yolo_sample_output_path}/labels/train/{image.strip('.png')}.txt')

    # Create PASCAL sample
    shutil.copyfile(file, f'{pascal_sample_output_path}/train/images/{image}')
    
with open(pascal_train_annotation_file, 'r') as f:
    annotations = json.load(f) 
sample_annotations = {k: v for k, v in annotations.items() if k in sample} 
with open(pascal_sample_output_path+"/train/annotations.json", "w") as f:
    json.dump(sample_annotations, f, indent=4)


print("PRE-PROCESSING SCRIPT COMPLETED", flush=True)
