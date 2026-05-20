"""
Functions for custom DR and FPR metrics when conventional inference is applied.

"""


# Import packages
import os
import shutil
import pandas as pd
import subprocess
import glob

"DR"
def detection_rate(model, version, exp, custom_configuration, inputpath, output_folder, conf=0.25, iou=0.7, save=True, plots=False):
    TP_dictionary = {'subject':[], 'lesion':[], 'subject_lesion':[], 'detected_at_least_once':[], 'nr_related_slices':[], 'nr_detections':[]}
    # Gather data to compute DR on
    test_images = os.listdir(inputpath+'/images/test') 
    test_images.sort()
    test_labels = os.listdir(inputpath+'/labels/test') # e.g. '89_1_top_1058_3.txt'
    test_labels.sort()

    # Define and initiate image placeholders
    image_placeholder = f'/wecare/home/lotte/Thesis/DATA/ETZ/evaluation_placeholder/images/test/'
    label_placeholder = f'/wecare/home/lotte/Thesis/DATA/ETZ/evaluation_placeholder/labels/test/'

    for image, label in zip(test_images, test_labels):
        # Re-instatiate the 'test' images and 'test' labels directories and copy in the current image and label
        shutil.rmtree(image_placeholder)
        os.makedirs(image_placeholder)
        shutil.copyfile(inputpath+'/images/test/'+image, image_placeholder+image)
        shutil.rmtree(label_placeholder)
        os.makedirs(label_placeholder)
        shutil.copyfile(inputpath+'/labels/test/'+label, label_placeholder+label)

        # Extract subject and lesion number
        subject_nr, take, _, slice_idx, lesion_nr = label.strip('.txt').split('_')
        subject_lesion = subject_nr+'_'+lesion_nr # e.g. 89_3

        # Run inference and extract results
        results = model.val(data=custom_configuration, split='test', project=output_folder+f'/{exp}/custom_runs', name=f'{subject_lesion}', conf=conf, iou=iou, plots=plots) 
        for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)
        TP = results.confusion_matrix.matrix[0,0] # amount of true positives for this image
        
        # Store results in dictionary
        if subject_lesion in TP_dictionary['subject_lesion']: # the lesion already came up before
            idx = TP_dictionary['subject_lesion'].index(subject_lesion) # get list index
            TP_dictionary['nr_related_slices'][idx] += 1
            if TP > 0: # a true positive was found so the lesion is detected (at least once)
                TP_dictionary['detected_at_least_once'][idx] = 1
                TP_dictionary['nr_detections'][idx] += 1
        else: # we haven't seen this lesion yet
            TP_dictionary['subject'].append(subject_nr)
            TP_dictionary['lesion'].append(lesion_nr)
            TP_dictionary['subject_lesion'].append(subject_lesion)
            TP_dictionary['detected_at_least_once'].append(1 if TP > 0 else 0) # if TP > 0, the lesion is detected (at least once)
            TP_dictionary['nr_related_slices'].append(1)
            TP_dictionary['nr_detections'].append(TP)  

        if save: # Save results to .csv file 
            TP_dataframe = pd.DataFrame(TP_dictionary)
            TP_dataframe.to_csv(output_folder+f"/{exp}/dr.csv")

    # Compute detection rate per lesion
    TP_dictionary['detection_rate_per_lesion'] = [x/y for x,y in zip(TP_dictionary['nr_detections'], TP_dictionary['nr_related_slices'])]
    if save: # Save results to .csv file and output custom metric
        TP_dataframe = pd.DataFrame(TP_dictionary)
        TP_dataframe.to_csv(output_folder+f"/{exp}/dr.csv")
        print(f"Saved custom detection rate results to {output_folder+f"/{exp}/dr.csv"}.", flush=True)
    custom_dr = sum(TP_dictionary['detected_at_least_once'])/len(TP_dictionary['detected_at_least_once'])
    return custom_dr


"FPR"
def false_positive_rate(model, image_inputpath, conf=0.25, iou=0.7):
    false_positive_images = []
    FP_counter = 0
    test_images = os.listdir(image_inputpath) 
    for image in test_images:
        # Run inference and extract results
        img_path = image_inputpath+'/'+image
        results = model.predict(img_path, conf=conf, iou=iou)
        for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)
        for pred in results:
            lesion_detected = list(pred.boxes.cls)
            if lesion_detected: # if True, a lesion was detected in an empty slice (i.e. there's a false positive)
                # print('A false prediction was made for:', image, flush=True)
                false_positive_images.append(image) 
                FP_counter += 1
    custom_fpr = FP_counter / len(test_images)
    return false_positive_images, custom_fpr
