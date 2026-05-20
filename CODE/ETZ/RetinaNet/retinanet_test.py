"""
Script to evaluate a RetinaNet model on the lesion detection task with conventional inference.

Computing: Precision, Recall, F1, mAP, DR and FPR.

"""

# Import libraries
import torch
from torchvision.ops import box_iou
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet
from retinanet_eval_loop import retinanet_evaluation
import pandas as pd
import os
import json
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

# Model to evaluate and inference parameters
exp = "SF"
conf_thresh = 0.1
iou_thresh = 0.55

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Results placeholder:
eval_results = {}
output_folder = "/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/evaluation_runs"

# Create and evaluate model 
model_path = f"/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_best_retinanet.pth" 
model = BoneRetinaNet(num_classes=2, score_thresh=conf_thresh, nms_thresh=iou_thresh, anchor_generator='small')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval() 


"""------------------------------------- Compute Precision, Recall, F1 and mAP scores ----------------------------------------"""
data_path = "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_Sample" # test datapath
[test_loader] = create_dataloaders(device, data_path, sets=["test"], batch_size=1) # test dataloader
precision, recall, f1, map50, map50_95 = retinanet_evaluation(model, test_loader, conf_thres=conf_thresh, iou_thres=iou_thresh) # evaluate
# Print and store results
print("\nPrecision:", precision, flush=True)
print("Recall:", recall, flush=True)
print("F1:", f1, flush=True)
print("mAP50:", map50)
print("mAP50-95:", map50_95)
eval_results['Box-P'] = [precision]
eval_results['Box-R'] = [recall]
eval_results['Box-F1'] = [f1]
eval_results['mAP50'] = [map50] 
eval_results['mAP50-95'] = [map50_95] 


"""--------------------------------------------- Compute Detection Rate score ------------------------------------------------"""
test_image_path = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/images/test/" # datapath
test_images = os.listdir(test_image_path)
test_target_path = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/annotations.json" # annotations
with open(test_target_path, 'r') as f:
        test_targets = json.load(f)

detection_model = AutoDetectionModel.from_pretrained(
            model_type="torchvision",
            model=model,
            device=device,
            confidence_threshold=conf_thresh,
            category_mapping={"1":"lesion"},
            load_at_init=True,
)  

TP_dictionary = {'subject':[], 'lesion':[], 'subject_lesion':[], 'detected_at_least_once':[], 'nr_related_slices':[], 'nr_detections':[]}
for test_image in tqdm(test_images):
        subject_nr, take, _, slice_idx, lesion_nr = test_image.strip('.png').split('_')
        subject_lesion = subject_nr+'_'+lesion_nr # e.g. 89_l3'

        prediction_results = get_prediction(test_image_path+test_image, detection_model) # run conventional inference
        
        # Compare predictions to ground truth 
        prediction_list = prediction_results.object_prediction_list
        predicted_boxes = [item.bbox.__dict__["box"] for item in prediction_list]
        pred_box_tensor = torch.tensor(predicted_boxes, dtype=torch.float32).to(device)
        target_boxes = test_targets[test_image]
        tgt_box_tensor = torch.tensor(target_boxes, dtype=torch.float32).to(device)
        TP = False
        if pred_box_tensor.size(0) > 0:
            ious = box_iou(pred_box_tensor, tgt_box_tensor)
            matched_gt = torch.zeros(tgt_box_tensor.size(0), dtype=torch.bool, device=device)
            for pred_iou in ious:
                max_iou, gt_idx = pred_iou.max(0)
                if max_iou >= 0.5 and not matched_gt[gt_idx]: # We found a true positive (i.e. the lesion is detected)!
                    TP = True
                    break

        # Store results in dictionary
        if subject_lesion in TP_dictionary['subject_lesion']: # the lesion already came up before
            idx = TP_dictionary['subject_lesion'].index(subject_lesion) # get list index
            TP_dictionary['nr_related_slices'][idx] += 1
            if TP:
                TP_dictionary['detected_at_least_once'][idx] = 1 # the lesion is successfully detected (at least once)
                TP_dictionary['nr_detections'][idx] += 1
        else: # we haven't seen this lesion yet
            TP_dictionary['subject'].append(subject_nr)
            TP_dictionary['lesion'].append(lesion_nr)
            TP_dictionary['subject_lesion'].append(subject_lesion)
            TP_dictionary['detected_at_least_once'].append(1 if TP else 0) # if TP = True, the lesion is successfully detected (at least once)
            TP_dictionary['nr_related_slices'].append(1)
            TP_dictionary['nr_detections'].append(TP)  

        # Save results to .csv file 
        TP_dataframe = pd.DataFrame(TP_dictionary)
        if not os.path.exists(output_folder+f"/{exp}"):
            os.makedirs(output_folder+f"/{exp}")
        TP_dataframe.to_csv(output_folder+f"/{exp}/dr.csv")

# Compute ratio per lesion and overall DR
TP_dictionary['detection_rate_per_lesion'] = [x/y for x,y in zip(TP_dictionary['nr_detections'], TP_dictionary['nr_related_slices'])]
TP_dataframe = pd.DataFrame(TP_dictionary)

# Store results
TP_dataframe.to_csv(output_folder+f"/{exp}/dr.csv")
print(f"Saved custom detection rate results to {output_folder+f"/{exp}/dr.csv"}.", flush=True)
custom_dr = sum(TP_dictionary['detected_at_least_once'])/len(TP_dictionary['detected_at_least_once'])
print("CUSTOM DR:", custom_dr)
eval_results['custom-dr'] = [custom_dr]



"""--------------------------------------------- Compute False Positive Rate -------------------------------------------------"""
test_image_path = "/wecare/home/lotte/Thesis/DATA/ETZ/negative_slice_subset/" # test datapath
test_images = os.listdir(test_image_path)

# Loop over negative (healthy, non-lesionous) images to check for false positives
fp = 0
for test_image in tqdm(test_images):
    prediction_results = get_prediction(test_image_path+test_image, detection_model)
    prediction_list = prediction_results.object_prediction_list
    predicted_boxes = [item.bbox.__dict__["box"] for item in prediction_list]
    pred_box_tensor = torch.tensor(predicted_boxes, dtype=torch.float32).to(device)
    if pred_box_tensor.size(0) > 0: # a lesion prediction was made in a non-lesion containing slice (false positive)
            fp += 1
    
# Compute and store FPR
custom_fpr = fp / len(test_images)
eval_results['custom_fpr'] = [custom_fpr]
print("CUSTOM FPR:", custom_fpr)

# Save all computed metrics
print("Saving evaluation metrics")
results_file = f'{output_folder}/{exp}_evaluation_metrics.csv'
results_df = pd.DataFrame(eval_results)
results_df.to_csv(results_file)
print("Saved evaluation metrics to:", results_file)


print("Completed conventional evaluation script.")