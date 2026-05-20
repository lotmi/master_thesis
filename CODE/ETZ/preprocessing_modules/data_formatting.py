"""
Script to put bounding boxes of the image slices into YOLO and/or PASCAL format

YOLO format:
1. folder called 'images' (optionally with 'train', 'validation', 'test' etc.) that contains:
    > your images (e.g. im1.png)
2. folder called 'labels' (optionally with 'train', 'validation', 'test' etc.) that contains:
    > text files with congruent names (e.g. im1.txt). One line per object in the image states: 
        > the class number 
        > the bounding box: x center, y center, width, height
        
PASCAL format:
one folder for each datasplit ('train', 'validation', 'test') that contains:
> a subfolder called 'images' with the images (e.g. im1.png)
> a annotations.json file containing a dictionary, with each entry showing:
    > an image name (e.g. im1.png) as key
    > the corresponding bounding boxes (x min, y min, x max, y max) as values 

"""

# Import packages
import os
import shutil
import pandas as pd
import json


""" YOLO Format """
def yolo_bbox(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3):
    # bbox as returned by regionprops: (min_row, min_col, max_row, max_col)
    width, height, y_min, x_min, y_max, x_max = im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    # bbox in YOLO format: (x_center, y_center, box_width, box_height)
    return [x_center, y_center, box_width, box_height]

def to_yolo_format(label_outputpath, propspath, datasplit, singular=False):
    # Re-instantiate label folder every time code is run 
    for split in datasplit.keys():
        shutil.rmtree(label_outputpath+"/"+split)
        os.makedirs(label_outputpath+"/"+split)

    # Get bounding boxes and other properties
    props_df = pd.read_csv(propspath, dtype={'scan_nr': str, 'take': int})
    for row, props in props_df.iterrows():
        row, scan_nr, take, view, slice_index, index, label, area, centroid_0, centroid_1, bbox_0, bbox_1, bbox_2, bbox_3, im_width, im_height = props.to_list()
        for split, subject_numbers in datasplit.items():
            if int(scan_nr) in subject_numbers:
                scan_split = split
                break
        
        # Convert bounding box
        x_center, y_center, box_width, box_height = yolo_bbox(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3)
            
        # Save converted bounding box to .txt file
        txt_filename = '{}_{}_{}_{}.txt'.format(scan_nr, take, view, slice_index)
        txt_path = label_outputpath+"/"+scan_split+"/"
        if txt_filename in os.listdir(txt_path) and not singular: # If text file already exists (this slice contains multiple lesions): append 
            file = open(txt_path+txt_filename, "a")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 0 = class label (lesion)
            file.close()
        elif singular:
            txt_filename = '{}_{}_{}_{}_n{}.txt'.format(scan_nr, take, view, slice_index, int(label))
            file = open(txt_path+txt_filename , "w")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) 
            file.close()
        else: # Else create a new text file 
            file = open(txt_path+txt_filename , "w")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) 
            file.close()
            
            
""" Pascal VOC Format """          
def to_pascal_format(pascal_path, propspath, datasplit):
    # create dictionaries with "image_name" : [[bounding box]] entries
    # so, data is e.g.: [ (img1, [[box1], [box2]]),  (img2, [box1]) ]
    train_data = {}
    val_data = {}
    test_data = {}
    data_dics = [train_data, val_data, test_data]
    
    # Get bounding boxes and other properties
    props_df = pd.read_csv(propspath, dtype={'scan_nr': str, 'take': int})
    for row, props in props_df.iterrows():
        # Immediately convert bounding box: y_min, x_min, y_max, x_max = bbox_0, bbox_1, bbox_2, bbox_3
        # bbox as returned by regionprops: (min_row, min_col, max_row, max_col)
        # bbox in pascal format: (x_min, y_min, x_max, y_max)
        row, scan_nr, take, view, slice_index, index, label, area, centroid_0, centroid_1, y_min, x_min, y_max, x_max, im_width, im_height = props.to_list()
        img_filename = '{}_{}_{}_{}.png'.format(scan_nr, take, view, slice_index)
        bbox = [x_min, y_min, x_max, y_max]
        
        for (split, subject_numbers), data_dic in zip(datasplit.items(), data_dics):
            if int(scan_nr) in subject_numbers:
                if img_filename in data_dic.keys(): # this image contains multiple bounding boxes: append to list of previously encountered boxes
                    data_dic[img_filename].append(bbox)
                else:
                    data_dic[img_filename] = [bbox] # this image is new: initiate in dictionary
                break
        
    # Save dictionaries as .json files
    with open(pascal_path+"/train/annotations.json", "w") as f:
            json.dump(train_data, f, indent=4)
    with open(pascal_path+"/val/annotations.json", "w") as f:
            json.dump(val_data, f, indent=4)        
    with open(pascal_path+"/test/annotations.json", "w") as f:
            json.dump(test_data, f, indent=4)
