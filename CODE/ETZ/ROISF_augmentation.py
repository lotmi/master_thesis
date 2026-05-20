"""
Script to perform training data augmentation with the adapted SF/SAHI slicing code.
Instead of using a fixed size sliding window, variable sized patches are created for lesionous bone tissue regions only.
Bone tissue is detected using the pre-trained YOLOv5 bone detector.
Outputted bounding boxes are cropped out of the original images and labels to extract patches.

For this purpose, the slicing code in the SAHI software was adjusted. 
Please consult the sahi/slicing.py script for the precise adaptations. 

An 'else' statement can be commented out to allow for negative samples (non-lesionous patches) as well.

"""

# Import packages
import os
import shutil
import json
import pandas as pd
import nibabel as nib
import skimage as skim 
from ultralytics import YOLO
from PIL import Image
import matplotlib.image
import subprocess
import glob


"""------------------------------------- Patch Creation --------------------------------------"""

# Access data and define goal paths
image_inputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_Sample/images/train/'
label_inputpath = '/wecare/projects/Slicer_ready_data/Original Patient Data/Uniform_scaling/NewLabels/'
slice_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_ROISF/images/train/'
props_outputpath = '/wecare/home/lotte/Thesis/CODE/ETZ/slice_properties_ROISF.csv'
yolo_outputpath = '/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_ROISF/labels'

# Initiate YOLO bone detector (with optimized inference hyperparameters)
iou = 0.5
conf = 0.393
image_names = os.listdir(image_inputpath)
bone_detection_weights = '/wecare/home/lotte/Thesis/CODE/ETZ/pretrained_weights/yolo5s.pt'
bone_detection_model = YOLO(bone_detection_weights) 

# Metadata placeholder:
props_dict = {'scan_nr':[], 'take':[], 'view':[], 'slice_index':[], 'patch_index':[], 'index':[], 'label':[], 'area': [], 'centroid-0': [], 'centroid-1': [], 'bbox-0': [], 'bbox-1': [], 'bbox-2': [], 'bbox-3': [], 'width':[], 'height':[]} 

# Go over detections and extract patches for bounding boxes that feature a bone lesion
subj_placeholder, take_placeholder = '', ''
emptyslices, lesionslices = 0, 0
for image_name in image_names:
    
    # Detect bone tissue regions in the image with YOLOv5 bone detector
    detections = bone_detection_model(image_inputpath+image_name, iou=iou, conf=conf)
    for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)

    # Loop over bone detections 
    for detection in detections:
        subj, take, _, slice_idx = image_name.strip('.png').split('_')
        if subj != subj_placeholder or take != take_placeholder:
            subj_placeholder, take_placeholder = subj, take
            lab_filename = f"CTP{subj}_00{take}.nii"
            label = nib.load(label_inputpath+lab_filename)
            label = skim.measure.label(label.get_fdata().astype(int))
        lab_slice = label[:,:,int(slice_idx)]

        image = Image.open(image_inputpath+image_name)
        box_tensor = detection.boxes.xyxy
        bboxes = box_tensor.cpu().numpy()
        for i, box in enumerate(bboxes):
            box = box.astype(int)
            xmin, ymin, xmax, ymax = box # bone bounding box
            label_patch = lab_slice[ymin:ymax, xmin:xmax] # corresponding label patch
            box = [xmin, ymin, xmax, ymax]
            image_patch = image.crop(box) # corresponding image patch

            # Check if the patch contains 
            info_table = pd.DataFrame(skim.measure.regionprops_table(label_patch, properties=['label', 'area', 'centroid', 'bbox'])).reset_index()
            if info_table.shape[0] > 0: # The patch contains a lesion: save patch and store metadata
                matplotlib.image.imsave(f'{slice_outputpath}/PATCH{i}_{subj}_{int(take)}_top_{slice_idx}.png', image_patch, cmap='gray')
                for _, props in info_table.iterrows():
                    lesionslices += 1
                    props_dict["scan_nr"].append(subj)
                    props_dict["take"].append(take)
                    props_dict["view"].append('top')
                    props_dict["slice_index"].append(slice_idx)
                    props_dict['patch_index'].append(i)
                    props_dict["height"].append(label_patch.shape[0])
                    props_dict["width"].append(label_patch.shape[1])
                    for prop, val in zip(props.keys(), props):
                        props_dict[prop].append(val)

            # else: # this is a negative patch
            #     matplotlib.image.imsave(f'{slice_outputpath}/PATCH{i}_{subj}_{int(take)}_top_{slice_idx}.png', image_patch, cmap='gray')
            #     txt_filename = 'PATCH{}_{}_{}_top_{}.txt'.format(i, subj, int(take), slice_idx)
            #     txt_path = yolo_outputpath+"/train/"
            #     file = open(txt_path+txt_filename , "w")
            #     file.write("") 
            #     file.close()  
            #     emptyslices += 1
            
print(f"Created a total of {emptyslices} empty slices and {lesionslices} slices with one or more lesions.")
# Copying Images (adjust paths if necessary)
shutil.copytree("/wecare/home/lotte/Thesis/DATA/ETZ/YOLO_ROISF/images/train", "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_ROISF/train/images", dirs_exist_ok=True)
print("Completed copying images from YOLO folder to PASCAL folder.")

# Save metadata
props_table = pd.DataFrame(props_dict)
props_table.to_csv(props_outputpath)
print(f"Saved lesion properties for all slices to {props_outputpath}.")


"""------------------------------------- Label Creation --------------------------------------"""

# (Re-)Access data
props_inputpath = '/wecare/home/lotte/Thesis/CODE/ETZ/slice_properties_ROISF.csv' 
props_df = pd.read_csv(props_inputpath, dtype={'scan_nr': str, 'take': int}) 
# props_df = props_table

# YOLO Conversion
from preprocessing_modules.data_formatting import yolo_bbox
print("Converting bounding boxes to YOLO format.", flush=True)
for row, props in props_df.iterrows():
    row, scan_nr, take, view, slice_index, patch_index, index, label, area, centroid_0, centroid_1, bbox_0, bbox_1, bbox_2, bbox_3, im_width, im_height = props.to_list()
    # Convert bounding box
    x_center, y_center, box_width, box_height = yolo_bbox(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3)
    # Save converted bounding box to .txt file
    txt_filename = 'PATCH{}_{}_{}_{}_{}.txt'.format(patch_index, scan_nr, take, view, slice_index)
    txt_path = yolo_outputpath+"/train/"
    if txt_filename in os.listdir(txt_path): # If text file already exists (this slice contains multiple bones): append 
        file = open(txt_path+txt_filename, "a")
        file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 0 = class label (bone)
        file.close()
    else: # Else create a new text file 
        file = open(txt_path+txt_filename , "w")
        file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 
        file.close()          
print("Completed YOLO bounding box conversion.", flush=True)

# PASCAL Conversion
pascal_outputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_ROISF/train/annotations.json"
with open(pascal_outputpath, 'r') as inpf:
    data_dic = json.load(inpf) 
print("Converting bounding boxes to PASCAL format.", flush=True)
for row, props in props_df.iterrows():
    row, scan_nr, take, view, slice_index, patch_index, index, label, area, centroid_0, centroid_1, y_min, x_min, y_max, x_max, im_width, im_height = props.to_list()
    img_filename = 'PATCH{}_{}_{}_top_{}.png'.format(patch_index, scan_nr, take, slice_index) #PATCH{i}_{subj}_{int(take)}_top_{slice_idx}.png'
    bbox = [x_min, y_min, x_max, y_max]
    if img_filename in data_dic.keys(): # this image contains multiple bounding boxes: append to list of previously encountered boxes
        data_dic[img_filename].append(bbox)
    else:
        data_dic[img_filename] = [bbox]
with open(pascal_outputpath, "w") as outf:
    json.dump(data_dic, outf, indent=4)  
print("Completed PASCAL bounding box conversion.", flush=True)


print("Completed label creation.")
