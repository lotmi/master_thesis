"""
Script to tune inference hyperparameters for a given YOLO model.

"""

# Import packages
from ultralytics import YOLO
import subprocess
import glob

# Set these yourself to determine which model version/edition is tuned:
v = "5" # 5 or 26
magnitude = 's' 

# Create model 
if v == "5":
    version = f'yolov{v}{size}' 
else:
    version = f'yolo{v}{size}'
configuration = '/home/u366836/thesis/CODE/TotalSegmentator/YOLO/yolo_config.yaml'
checkpoint = "/home/u366836/thesis/CODE/TotalSegmentator/YOLO/runs/detect/yolo26s_train/weights/best.pt"
model = YOLO(checkpoint)

# Create parameter grid (best confidence thresholds can be read from the images outputted by Ultralytics)
ious = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Run grid search
best_iou = 0
best_weighted_f1 = 0
best_run = 0
output_folder = "/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/inference_params"
run_counter = 1
for iou in ious:
    results = model.val(data=configuration, split='val', name=f'{version}_test', visualize=True, iou=iou)
    for path in glob.glob("/tmp/pymp*"):
        subprocess.run(["rm", "-r", path], capture_output=True)
    results_df = results.to_df()
    weighted_f1 = 0.25*results_df.item(0, "Box-P") + 0.75*results_df.item(0, "Box-R")
    if weighted_f1 > best_weighted_f1:
        best_weighted_f1 = weighted_f1
        best_iou = iou
        best_run = run_counter
        print(f"Saved improved iou ({best_iou}) at run {run_counter} for an F1 of {best_weighted_f1}", flush=True)
    run_counter += 1

print(f"Best iou is {best_iou} at run {best_run} for an F1 of {best_weighted_f1}", flush=True)