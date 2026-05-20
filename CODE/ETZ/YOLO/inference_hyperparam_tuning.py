"""
Script to tune the hyperparameters for model evaluation/inference:
- Confidence threshold (def. = 0.25)
- IoU for NMS (def. = 0.7)

Metric to maximize: weighted F1 (trading off both precision (0.25) and recall (0.75))

"""

# Import packages
from ultralytics import YOLO
import subprocess
import glob

def compute_weighted_f1(precision, recall, pw=0.25, rw=0.75):
    if pw + rw != 1.0:
        print("Weights for precision and recall do not add up to 1. Returning 0 for weighted F1 and continuing...")
        return 0
    else:
        weighted_f1 = pw*precision + rw*recall
        return weighted_f1

# Set these variables yourself to indicate for which model the best-performing inference parameters should be found
version = "5"
exp = "ROISF"

# Initiate model and data configuration
model_name = f"YOLOv{version}/{exp}"
weights_path = f"/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/{model_name}_train/weights/best.pt"
model = YOLO(weights_path)
configuration = f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_config_{exp}.yaml'

# Parameter Grid and Grid Search Loop
# PLEASE READ THE OPTIMAL CONFIDENCE THRESHOLD (IN TERMS OF F1 FROM THE AUTOMATICALLY CREATED PLOT BY YOLO)
# confs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ious = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# best_conf = 0
best_iou = 0
best_weighted_f1 = 0
best_run = 0
output_folder = "/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/hyperparameters/inference_hyperparam_runs"
run_counter = 1
# for conf in tqdm(confs):
for iou in ious:
    results = model.val(data=configuration, split='val', project=output_folder, visualize=False, iou=iou) # conf=conf
    for path in glob.glob("/tmp/pymp*"):
        subprocess.run(['rm', '-r', path], capture_output=True)
    results_df = results.to_df()
    weighted_f1 = compute_weighted_f1(results_df.item(0, "Box-P"), results_df.item(0, "Box-R"))
    if weighted_f1 > best_weighted_f1:
        best_weighted_f1 = weighted_f1
        # best_conf = conf
        best_iou = iou
        best_run = run_counter
        print(f"Saved improved iou inference parameter for YOLO at an F1 score of {best_weighted_f1}: iou = {best_iou}")
    run_counter += 1
    
print(f"Best inference parameter for YOLO found at run {best_run}: iou = {best_iou} gives a F1 score of {best_weighted_f1}. Check the plots for optimal confidence threshold.")