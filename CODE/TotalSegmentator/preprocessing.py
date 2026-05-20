"""
This script presents the following pre-processing pipeline for Kahler CT scans: 

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
# This gives set magnitudes of 54 (4612 slices), 12 (844 slices) and 12 (930 slices) respectively 
# Note that the data exploration initially indicated there would be more slices because it was not taken into account that some lesions are featured the same slice!
# datasplit = {"train":(0, 73), "val":(74, 88), "test":(89, 103)}

datasplit = {"train":(2, 4, 7, 100, 15, 21, 22, 24, 26, 27, 28,
                      29, 31, 32, 33, 34, 36, 37, 40, 41, 42, 43,
                      46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 
                      60, 62, 63, 64, 65, 66, 67, 69, 71, 73, 74,
                      75, 76, 77, 78, 79, 86, 89, 91, 92, 93, 94, 
                      95, 97, 99), 
             "val":(20, 38, 39, 44, 59, 61, 68, 70, 82, 85), 
             "test":(102, 10, 12, 35, 45, 81, 83, 87, 90, 98)}


# """------------------------------------- Scaling (step 1 + 2) --------------------------------------"""
# from preprocessing_modules.scaling import apply_window, apply_normalization

# # Access data 
# inputpath = '/wecare/projects/Slicer_ready_data/Original Patient Data/Uniform_scaling/Images'
# outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/Scaled_Scans'
# all_images = sorted(os.listdir(inputpath))

# # Loop over and process all subjects
# print("Scaling all CT scans.", flush=True)
# for img_file in all_images:
#     scan_nr = img_file[0:9]
#     img = nib.load(inputpath+'/'+img_file)
#     img_arr = img.get_fdata()
#     windowed_img = apply_window(img_arr, W=1800, L=400) # to check: print(windowed_img.min(), windowed_img.max(), flush=True) # should be between -500 and 1300
#     normalized_img = apply_normalization(windowed_img) # to check: print(normalized_img.min(), normalized_img.max(), flush=True) # should be between 0 and 1
#     np.save(f"{outputpath}/{scan_nr}_scaled.npy", normalized_img)
# print("Completed scaling of all CT scans.", flush=True)

    
    
# """------------------------------------- Slicing (step 3) -----------------------------------------"""
# from preprocessing_modules.slicing import create_slices 

# # Access data
# image_inputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/Scaled_Scans'
# label_inputpath = '/wecare/projects/Slicer_ready_data/Original Patient Data/Uniform_scaling/NewLabels'
# slice_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_cross/images'
# props_outputpath = '/wecare/home/lotte/Thesis/CODE/ETZ/cross_slice_properties.csv' 
# all_labels = sorted(os.listdir(label_inputpath))
# scan_nrs = [f[0:9] if not "100" in f and not "102" in f else f[0:10] for f in all_labels]
# scan_ints = [f[3:5] if not "100" in f and not "102" in f else f[3:6] for f in all_labels]
# takes = [f[6:9] if not "100" in f and not "102" in f else f[7:10] for f in all_labels]

# # Define desired slice properties
# properties = ['label', 'area', 'centroid', 'bbox'] 
# props_dict = {'scan_nr':[], 'take':[], 'view':[], 'slice_index':[], 'index':[], 'label':[], 'area': [], 'centroid-0': [], 'centroid-1': [], 'bbox-0': [], 'bbox-1': [], 'bbox-2': [], 'bbox-3': [], 'width':[], 'height':[]} 
# views = ['top']

# # Loop over and process all scans
# print("Creating {} slices for all CT scans.".format("& ".join(views)), flush=True)
# for lab_file, scan_nr, scan_int, take in zip(all_labels, scan_nrs, scan_ints, takes):
#     if scan_nr not in lab_file:
#         print(f"ISSUE: mismatching scan number ({scan_nr}) and label file ({lab_file}).")
#         break
#     label_nifti = nib.load(label_inputpath+'/'+lab_file)
#     label = skim.measure.label(label_nifti.get_fdata().astype(int))
#     img = np.load(f"{image_inputpath}/{scan_nr}_scaled.npy")
#     props_dict = create_slices(scan_int, take, img, label, props_dict, datasplit, slice_outputpath, views, properties)
# print("Completed slicing all CT scans.", flush=True)    

# # Save properties
# props_table = pd.DataFrame(props_dict)
# props_table.to_csv(props_outputpath)
# print(f"Saved lesion properties for all slices to {props_outputpath}.", flush=True)


    
"""------------------------------------- B-Box Conversion (step 4) ---------------------------------"""
from preprocessing_modules.data_formatting import to_yolo_format, to_pascal_format

# Access data
props_inputpath = '/wecare/home/lotte/Thesis/CODE/ETZ/cross_slice_properties.csv'  

# # YOLO Conversion
# yolo_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_cross/labels'
# print("Converting bounding boxes to YOLO format.", flush=True)
# to_yolo_format(yolo_outputpath, props_inputpath, datasplit)
# print("Completed YOLO bounding box conversion.", flush=True)

# PASCAL Conversion
pascal_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_cross_SF"
print("Converting bounding boxes to PASCAL format.", flush=True)
to_pascal_format(pascal_outputpath, props_inputpath, datasplit)
print("Completed PASCAL bounding box conversion.", flush=True)


# """----------------------------- Copying Images (not generalizeable) -------------------------------"""
# # Change paths to what you want
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_SF/images/train", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_SF/train/images", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/images/train", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/train/images", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/images/val", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/val/images", dirs_exist_ok=True)


# """------------------------ Sampling Images (to fix data imbalance) -------------------------------"""

# train_sample = pd.read_csv('/wecare/home/lotte/Thesis/CODE/ETZ/cross_sampled_data_properties.csv', dtype={'scan_nr': str, 'take': int})
# selected_train_images = [f'{row['scan_nr']}_{row['take']}_{row['view']}_{row['slice_index']}.png' for index, row in train_sample.iterrows()]
# selected_train_labels = [f'{row['scan_nr']}_{row['take']}_{row['view']}_{row['slice_index']}.txt' for index, row in train_sample.iterrows()]

# # YOLO
# yolo_src_pth = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_cross/'
# yolo_dst_pth = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_cross_sampled/'
# for image, label in zip(selected_train_images, selected_train_labels):
#     shutil.copyfile(yolo_src_pth+'images/train/'+image, yolo_dst_pth+'images/train/'+image)
#     shutil.copyfile(yolo_src_pth+'labels/train/'+label, yolo_dst_pth+'labels/train/'+label)

# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/images/test", "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sample/images/test", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/images/val", "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sample/images/val", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/labels/test", "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sample/labels/test", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/labels/val", "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sample/labels/val", dirs_exist_ok=True)

# # PASCAL
# pascal_src_pth = '/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/train/'
# pascal_dst_pth = '/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_Sample/train/'
# for image in selected_train_images:
#     shutil.copyfile(pascal_src_pth+'images/'+image, pascal_dst_pth+'images/'+image)
# with open('/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/train/annotations.json', 'r') as f:
#     annotations = json.load(f) 
# sample_annotations = {k: v for k, v in annotations.items() if k in selected_train_images} 
# with open(pascal_dst_pth+"annotations.json", "w") as f:
#     json.dump(sample_annotations, f, indent=4)

# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/test", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_Sample/test", dirs_exist_ok=True)
# shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/val", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_Sample/val", dirs_exist_ok=True)


print("PRE-PROCESSING SCRIPT COMPLETED", flush=True)
