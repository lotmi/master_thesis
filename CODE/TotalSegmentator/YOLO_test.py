"""
Script to run inference with / evaluate a YOLO Model. 

Standard performance metrics (P, R, F1, mAP) are computed. 

"""

# Import packages
from ultralytics import YOLO
import subprocess
import glob 

# Set these yourself to determine which model version/edition is tuned:
v = "5" # 5 or 26
magnitude = 's' 
iou = 0.5 # optimized inference param
conf = 0.393 # optimized inference param

# Evaluate
if v == "5":
    version = f'yolov{v}{size}' 
else:
    version = f'yolo{v}{size}'
configuration = '/home/u366836/thesis/CODE/TotalSegmentator/YOLO/yolo_config.yaml'
checkpoint = f"/home/u366836/thesis/CODE/TotalSegmentator/YOLO/runs/detect/{version}_train/weights/best.pt"
model = YOLO(checkpoint)
test_results = model.val(data=configuration, split='test', name=f'{version}_test', visualize=True, iou=iou, conf=conf)
results_df = test_results.to_df()
print(results_df, flush=True)