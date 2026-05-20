"""
Script to evaluate a RetinaNet model on the lesion detection task with (ROI-)SAHI.

Computing: Precision, Recall, F1, mAP, DR and FPR.

"""

# Import libraries
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import os
import torch
import json
from tqdm import tqdm
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
from torchvision.ops import box_iou
from retinanet_model import BoneRetinaNet


# Set these variables yourself to indicate which model should be test (and with which inference parameters)
exp = "boneboxes"
model_path = f"/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_best_retinanet.pth"
conf = 0.25
iou = 0.5
ROI_informed = True

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create and evaluate model 
model = BoneRetinaNet(num_classes=2, score_thresh=conf, nms_thresh=iou, anchor_generator='small')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()    
detection_model = AutoDetectionModel.from_pretrained(
            model_type="torchvision",
            model=model,
            device=device,
            confidence_threshold=conf,
            category_mapping={"1":"lesion"},
            load_at_init=True,
)    


# Additional (ROI-)SAHI parameters
slice_size = 128
overlap_ratio = 0.2
bone_detection_model = YOLO("/wecare/home/lotte/Thesis/CODE/ETZ/pretrained_weights/yolo5s.pt")

# Create folder and path for evaluation output
output_folder = "/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/evaluation_runs"
if ROI_informed:
    results_file = f"{output_folder}/{exp}_ROISAHI_evaluation_metrics.csv"
else:
    results_file = f"{output_folder}/{exp}_SAHI_evaluation_metrics.csv"

# Results placeholder
results = {}


"""------------------------------------- Compute Precision, Recall, F1 and mAP scores ----------------------------------------"""
if ROI_informed:
    sliced_test_vispath = output_folder+f"/{exp}_ROISAHI/visualizations/"
else:
    sliced_test_vispath = output_folder+f"/{exp}_SAHI/visualizations/"
if not os.path.exists(sliced_test_vispath):
    os.makedirs(sliced_test_vispath)

test_image_path = "/wecare/home/lotte/Thesis/DATA/ETZ/YOLO/images/test/" # test datapath
test_images = os.listdir(test_image_path)
test_target_path = "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL/test/annotations.json" # test labelpath
with open(test_target_path, 'r') as f:
    test_targets = json.load(f)

tp, fp, fn = 0, 0, 0
all_preds = [] # placeholder for mAP computation
all_targets = [] # placeholder for mAP computation
# Loop over test images to get slicing-aided inference results
for test_image in tqdm(test_images):
    if ROI_informed:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            ROI_informed=True, bone_detection_model=bone_detection_model,
                                            postprocess_match_threshold=iou, verbose=0)  
    else:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            slice_height=slice_size, slice_width=slice_size, 
                                            overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
                                            postprocess_match_threshold=iou, verbose=0)
    sahi_results.export_visuals(export_dir=sliced_test_vispath, file_name=test_image)
    
    # Compare inference results with ground truths
    prediction_list = sahi_results.object_prediction_list
    predicted_boxes = [item.bbox.__dict__["box"] for item in prediction_list]
    pred_box_tensor = torch.tensor(predicted_boxes, dtype=torch.float32).to(device)
    pred_scores = [item.score.__dict__["value"] for item in prediction_list]
    pred_score_tensor = torch.tensor(pred_scores, dtype=torch.float32).to(device)
    map_preds = {'boxes':pred_box_tensor, 'scores':pred_score_tensor, 'labels':torch.ones((pred_box_tensor.shape[0],), dtype=torch.int64).to(device)}
    all_preds.append(map_preds)
    target_boxes = test_targets[test_image]
    tgt_box_tensor = torch.tensor(target_boxes, dtype=torch.float32).to(device)
    map_tgts = {"boxes":tgt_box_tensor, "labels":torch.ones((tgt_box_tensor.shape[0],), dtype=torch.int64).to(device)}
    all_targets.append(map_tgts)
    if pred_box_tensor.size(0) == 0:
            fn += tgt_box_tensor.size(0)
            continue
    if tgt_box_tensor.size(0) == 0:
            fp += pred_box_tensor.size(0)
    ious = box_iou(pred_box_tensor, tgt_box_tensor)
    matched_gt = torch.zeros(tgt_box_tensor.size(0), dtype=torch.bool, device=device)
    for pred_iou in ious:
        max_iou, gt_idx = pred_iou.max(0)
        if max_iou >= 0.5 and not matched_gt[gt_idx]:
            tp += 1
            matched_gt[gt_idx] = True
        else:
            fp += 1
    fn += (~matched_gt).sum().item()

# Compute metrics
if tp + fp + fn > 0:
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
else:
    precision = recall = f1 = 0.0
map_metric = MeanAveragePrecision() # iou_thresholds=None equals the default 0.5-0.95 (0.05 steps) range
map_metric.update(all_preds, all_targets)
map_results = map_metric.compute()
map50 = map_results['map_50'].item()
map50_95 = map_results['map'].item()

# Store metrics
results['Box-P'] = [precision]
results['Box-R'] = [recall]
results['Box-F1'] = [f1]
results['mAP50'] = [map50] 
results['mAP50-95'] = [map50_95] 
print("PRECISION:", precision)
print("RECALL:", recall)
print("F1:", f1)
print("MAP50:", map50)
print("MAP50-95:", map50_95)

# Intermediate save:
results_df = pd.DataFrame(results)
results_df.to_csv(results_file)


"""--------------------------------------------- Compute Detection Rate score ------------------------------------------------"""
test_image_path = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/images/test/" # test datapath
test_images = os.listdir(test_image_path)
test_target_path = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars/annotations.json" # test labelpath
with open(test_target_path, 'r') as f:
    test_targets = json.load(f)

# Go over all test images to check for correctly detected lesions
TP_dictionary = {'subject':[], 'lesion':[], 'subject_lesion':[], 'detected_at_least_once':[], 'nr_related_slices':[], 'nr_detections':[]}
for test_image in tqdm(test_images):
    subject_nr, take, _, slice_idx, lesion_nr = test_image.strip('.png').split('_')
    subject_lesion = subject_nr+'_'+lesion_nr # e.g. 89_l3
    if ROI_informed:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            ROI_informed=True, bone_detection_model=bone_detection_model,
                                            postprocess_match_threshold=iou, verbose=0)  
    else:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            slice_height=slice_size, slice_width=slice_size, 
                                            overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
                                            postprocess_match_threshold=iou, verbose=0)
    
    # Compare inference results to ground truths
    prediction_list = sahi_results.object_prediction_list
    predicted_boxes = [item.bbox.__dict__["box"] for item in prediction_list]
    pred_box_tensor = torch.tensor(predicted_boxes, dtype=torch.float32).to(device)
    target_boxes = test_targets[test_image]
    tgt_box_tensor = torch.tensor(target_boxes, dtype=torch.float32).to(device)
    if pred_box_tensor.size(0) == 0:
            continue
    ious = box_iou(pred_box_tensor, tgt_box_tensor)
    matched_gt = torch.zeros(tgt_box_tensor.size(0), dtype=torch.bool, device=device)
    TP = False
    for pred_iou in ious:
        max_iou, gt_idx = pred_iou.max(0)
        if max_iou >= 0.5 and not matched_gt[gt_idx]: # We found a True Positive Detection!
            TP = True
            matched_gt[gt_idx] = True
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


# Compute ratio per lesion and overall DR
TP_dictionary['detection_rate_per_lesion'] = [x/y for x,y in zip(TP_dictionary['nr_detections'], TP_dictionary['nr_related_slices'])]
custom_dr = sum(TP_dictionary['detected_at_least_once'])/len(TP_dictionary['detected_at_least_once'])


# Store results
results['custom_dr'] = [custom_dr]
print("CUSTOM DR:", custom_dr)
TP_dataframe = pd.DataFrame(TP_dictionary)
if ROI_informed:
    if not os.path.exists(output_folder+f"/{exp}_ROISAHI"):
        os.makedirs(output_folder+f"/{exp}_ROISAHI")
    TP_dataframe.to_csv(output_folder+f"/{exp}_ROISAHI/dr.csv")
else:
    if not os.path.exists(output_folder+f"/{exp}_SAHI"):
        os.makedirs(output_folder+f"/{exp}_SAHI")
    TP_dataframe.to_csv(output_folder+f"/{exp}_SAHI/dr.csv")


"""--------------------------------------------- Compute False Positive Rate -------------------------------------------------"""
test_image_path = "/wecare/home/lotte/Thesis/DATA/ETZ/negative_slice_subset/" # test datapath
test_images = os.listdir(test_image_path)

# Loop over negative (healthy, non-lesionous) images to check for false positives
fp = 0
for test_image in tqdm(test_images):
    if ROI_informed:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            ROI_informed=True, bone_detection_model=bone_detection_model,
                                            postprocess_match_threshold=iou, verbose=0)  
    else:
        sahi_results = get_sliced_prediction(test_image_path+test_image , detection_model, 
                                            slice_height=slice_size, slice_width=slice_size, 
                                            overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
                                            postprocess_match_threshold=iou, verbose=0)
    
    prediction_list = sahi_results.object_prediction_list
    predicted_boxes = [item.bbox.__dict__["box"] for item in prediction_list]
    pred_box_tensor = torch.tensor(predicted_boxes, dtype=torch.float32).to(device)
    if pred_box_tensor.size(0) > 0: # a lesion prediction was made in a non-lesion containing slice (false positive)
            fp += 1
    
# Compute and store FPR 
custom_fpr = fp / len(test_images)
results['custom_fpr'] = [custom_fpr]
print("CUSTOM FPR:", custom_fpr)


# Save all computed metrics
print("Saving evaluation metrics")
results_df = pd.DataFrame(results)
results_df.to_csv(results_file)
print("Saved evaluation metrics to:", results_file)

print("Completed SAHI evaluation script.")
