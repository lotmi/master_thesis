"""
Script to create 2D slices of the ETZ CT scans.

The code below loops over all CT slices and extracts those that display a lesion. 
The extracted CT slices are saved in an 'images' folder as .png files.
Metadata (area, bounding box, centroid, etc.) is saved in a 'properties.csv' file.

"""

# Import packages
import os
import pandas as pd
import matplotlib.image
import skimage as skim 


def compute_properties(lab_slice, props_dict, scan_nr, take, view, i, properties = ['label', 'area', 'centroid', 'bbox']):
    info_table = pd.DataFrame(skim.measure.regionprops_table(lab_slice, properties=properties)).reset_index()
    lesion_present = False
    lesion_amount = info_table.shape[0]
    labels = []
    if lesion_amount > 0: # there is a lesion in this slice!
        for _, props in info_table.iterrows():
            lesion_present = True
            props_dict["scan_nr"].append(scan_nr)
            props_dict["take"].append(take)
            props_dict["view"].append(view)
            props_dict["slice_index"].append(i)
            props_dict["height"].append(lab_slice.shape[0])
            props_dict["width"].append(lab_slice.shape[1])
            for prop, val in zip(props.keys(), props):
                props_dict[prop].append(val)
            labels.append(props['label'])
    return props_dict, lesion_present, labels


def save_slice(scan_nr, take, view, i, ct_slice, datasplit, path, labels, singular): 
    # Datasplit determines in which set (train, val or test) a subject should be putted
    scan_idx = int(scan_nr)
    found = False
    for split, subject_numbers in datasplit.items():
        if scan_idx in subject_numbers:
            found = True
            if not singular:
                matplotlib.image.imsave(f'{path}/{split}/{scan_nr}_{int(take)}_{view}_{i}.png', ct_slice, cmap='gray')
            elif singular:
                for lab in labels:
                    matplotlib.image.imsave(f'{path}/{split}/{scan_nr}_{int(take)}_{view}_{i}_l{int(lab)}.png', ct_slice, cmap='gray')
            break
    if not found:
        print("Scan number not found in split:", scan_idx)


def create_slices(scan_nr, take, img, label, props_dict, datasplit, path, views=['side','front','top'], properties = ['label', 'area', 'centroid', 'bbox'], singular=False, img_save=True):
    views_dims = {'side':0, 'front':1, 'top':2} # side = x = sagittal, front = y = coronal, top = z = axial
    dims = [views_dims[x] for x in views]
    for dim, view in zip(dims, views): 
        for i in range(0, label.shape[dim]): # loop over the indices to generate all possible slices 
            if dim == 0:
                lab_slice = label[i,:,:]
                ct_slice = img[i,:,:]
            elif dim == 1:
                lab_slice = label[:,i,:]
                ct_slice = img[:,i,:]
            elif dim == 2:
                lab_slice = label[:,:,i]
                ct_slice = img[:,:,i]
                
            # Compute regionprops for a given slice and save lesion containing slices
            props_dict, lesion_present, labels = compute_properties(lab_slice, props_dict, scan_nr, take, view, i, properties)
            if lesion_present and img_save:   
                save_slice(scan_nr, take, view, i, ct_slice, datasplit, path, labels, singular)         
    return props_dict
