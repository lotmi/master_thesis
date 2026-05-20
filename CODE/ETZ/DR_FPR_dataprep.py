"""
Create sets of singular slices and negative slices for the computation of the custom DR and FPR metrics respectively.

"""

# Import packages
import os
import shutil
import random
import json
import pandas as pd
import nibabel as nib
import matplotlib.image
from preprocessing_modules.data_formatting import yolo_bbox
from preprocessing_modules.scaling import apply_window, apply_normalization
from tqdm import tqdm
import random


"""------------------------------------- Singular Slices --------------------------------------"""

# Access data
image_inputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sampled/images/test/"
all_images = os.listdir(image_inputpath)    
all_images.sort()
image_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/images/test/"
label_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/labels/test/"
pascal_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/annotations.json"

# Loop over all images to create singular slices
slice_properties = pd.read_csv('/wecare/home/lotte/Thesis/CODE/ETZ/slice_properties.csv', dtype={'scan_nr': str, 'take': int, 'slice_index': int})
print("Creating singular slices")
singular_annotations = {}
err = False
for image in all_images:
    if err:
        break
    img_subject_nr, img_take, _, img_slice_idx = image.strip('.png').split('_')
    sub_table = slice_properties[slice_properties["scan_nr"]==img_subject_nr][slice_properties["take"]==int(img_take)][slice_properties["slice_index"]==int(img_slice_idx)]
    
    for row, props in sub_table.iterrows(): # For each lesion in an image: retrieve lesion information and create an own (singular) slice + annotation
        row, scan_nr, take, view, slice_index, index, label, area, centroid_0, centroid_1, bbox_0, bbox_1, bbox_2, bbox_3, im_width, im_height = props.to_list()

        # Create YOLO label
        x_center, y_center, box_width, box_height = yolo_bbox(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3) # Convert bounding box
        txt_filename = '{}_{}_{}_{}_l{}.txt'.format(scan_nr, take, view, slice_index, int(label))
        file = open(label_outputpath+txt_filename , "w")
        file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) 
        file.close()

        # Create PASCAL annotations
        img_filename = '{}_{}_{}_{}_l{}.png'.format(scan_nr, take, view, slice_index, int(label))
        y_min, x_min, y_max, x_max = bbox_0, bbox_1, bbox_2, bbox_3
        bbox = [x_min, y_min, x_max, y_max]
        
        if img_filename in singular_annotations.keys(): 
            print("FOUND DUPLICATE SINGULAR IMAGE:", img_filename)
            err = True
            break
        else:
            singular_annotations[img_filename] = [bbox] # this image is new: initiate in dictionary
        
        # Copy image 
        shutil.copyfile(image_inputpath+image, image_outputpath+img_filename)

# Save dictionaries as .json files
with open(pascal_outputpath, "w") as f:
            json.dump(singular_annotations, f, indent=4)

print("Completed creating singular slices")



"""------------------------------------- Negative Slices --------------------------------------"""

# Access healthy patient data
image_inputpath = "/wecare/projects/Slicer_ready_data/Healthy_patients/Healthy_patients_reader_study/Nifti"
slice_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/negative_slices"

# Pre-process healthy patient data
all_images = os.listdir(image_inputpath)
print("Scaling and slicing negative CT scans:", flush=True)
for img_file in tqdm(all_images):
    scan_nr = img_file[0:10]
    img = nib.load(image_inputpath+'/'+img_file)
    img_arr = img.get_fdata()
    windowed_img = apply_window(img_arr, W=1800, L=400)
    normalized_img = apply_normalization(windowed_img) 
    for i in range(0, normalized_img.shape[2]): 
        ct_slice = normalized_img[:,:,i]
        matplotlib.image.imsave(f'{slice_outputpath}/HEALTHY_{scan_nr}_{i}.png', ct_slice, cmap='gray')

# Sample negative (healthy, non-lesionous) slices
random.seed(42)
output_path = '/wecare/home/lotte/Thesis/DATA/ETZ/negative_slice_subset/'
input_path = '/wecare/home/lotte/Thesis/DATA/ETZ/negative_slices/'

negative_images = sorted(os.listdir(input_path))
sample_ratio = 0.10
sample_size = int(sample_ratio * len(negative_images)) # 951 slices, similar to positive test set

print(f"Creating sample (size={sample_size}) for FPR computation")
sample = random.sample(negative_images, sample_size)
for image in sample:
    file = f'{input_path}{image}'
    shutil.copyfile(file, f'{output_path}{image}')

print("Completed creating negative (healthy, non-lesionous) slices")