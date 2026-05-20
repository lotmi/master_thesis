"""
Script to perform training data augmentation with the original SF/SAHI slicing code.

"""

# Import packages
import os
import json
import pandas as pd
import nibabel as nib
import skimage as skim 
from tqdm import tqdm

from preprocessing_modules.data_formatting import yolo_bbox
from sahi.slicing import slice_image

# Access data and define goal paths
image_inputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sampled/images/train/'
label_inputpath = '/wecare/projects/Slicer_ready_data/Original Patient Data/Uniform_scaling/NewLabels/'
slice_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_SF/images/train/'
props_outputpath = '/wecare/home/lotte/Thesis/CODE/ETZ/slice_properties_SF.csv' 
image_names = os.listdir(image_inputpath)

yolo_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_SF/labels/train/'
pascal_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_SF/train/annotations.json'
with open(pascal_outputpath, 'r') as inpf:
    pascal_data_dic = json.load(inpf) 

# Go over images and extract patches with a 128 x 128 sliding window
emptyslices, lesionslices = 0, 0
subj_placeholder, take_placeholder = '', ''
for image_name in tqdm(image_names):

    subj, take, _, slice_idx = image_name.strip('.png').split('_')
    slice_name_base = f'PATCH_{subj}_{int(take)}_top_{slice_idx}' # sahi code will turn this into {slice_name}_{slice_suffixes}{suffix}
    # e.g. PATCH_63_1_top_656_336_260_400_324.png
    
    # Retrieve sliding window patch coordinates
    sliced_image_result, slice_bboxes = slice_image(image_inputpath+image_name, slice_height=128, slice_width=128, 
                                                    output_file_name=slice_name_base, output_dir=slice_outputpath, return_bboxes=True)

    if subj != subj_placeholder or take != take_placeholder:
        subj_placeholder, take_placeholder = subj, take
        lab_filename = f"CTP{subj}_00{take}.nii"
        label = nib.load(label_inputpath+lab_filename)
        label = skim.measure.label(label.get_fdata().astype(int))
    lab_slice = label[:,:,int(slice_idx)]

    # Use patch coordinates to slice patches from the original image and labels
    for bbox in slice_bboxes: 
        xmin, ymin, xmax, ymax = bbox
        label_patch = lab_slice[ymin:ymax, xmin:xmax]
        im_width, im_height = label_patch.shape[1], label_patch.shape[0]

        yolo_txtname = f'PATCH_{subj}_{int(take)}_top_{slice_idx}_{xmin}_{ymin}_{xmax}_{ymax}.txt'
        pascal_slicename = f'PATCH_{subj}_{int(take)}_top_{slice_idx}_{xmin}_{ymin}_{xmax}_{ymax}.png'

        info_table = pd.DataFrame(skim.measure.regionprops_table(label_patch, properties=['label', 'area', 'centroid', 'bbox'])).reset_index()
        if info_table.shape[0] > 0: # lesion-containing patch; save lesion info
            lesionslices+=1
            for _, props in info_table.iterrows():
                bbox_0, bbox_1, bbox_2, bbox_3 = props['bbox-0'], props['bbox-1'], props['bbox-2'], props['bbox-3'] # y_min, x_min, y_max, x_max

                # Pascal
                pascal_box = [bbox_1, bbox_0, bbox_3, bbox_2]
                if pascal_slicename in pascal_data_dic.keys(): 
                    pascal_data_dic[pascal_slicename].append(pascal_box)
                else:
                    pascal_data_dic[pascal_slicename] = [pascal_box]

                # YOLO
                x_center, y_center, box_width, box_height = yolo_bbox(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3)
                if yolo_txtname in os.listdir(yolo_outputpath): 
                    file = open(yolo_outputpath+yolo_txtname, "a")
                    file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) 
                    file.close()
                else: 
                    file = open(yolo_outputpath+yolo_txtname , "w")
                    file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) 
                    file.close()  
                    
        else: # non-lesionous patch; create empty annotations
            emptyslices+=1

            file = open(yolo_outputpath+yolo_txtname , "w")
            file.write("")
            file.close()  

            pascal_data_dic[pascal_slicename] = []


with open(pascal_outputpath, "w") as outf:
    json.dump(pascal_data_dic, outf, indent=4)  

print("Completed SF augmentation.")
print(f"Created a total of {emptyslices} empty slices and {lesionslices} slices with one or more lesions.")
