"""
Script to tune the YOLO bone detection model (either version 5 or 26).

"""

# Import packages
from ultralytics import YOLO

# Set these yourself to determine which model version/edition is tuned:
v = "5" # 5 or 26
magnitude = 's' 

# Load model 
if v == "5":
    version = f'yolov{v}{size}.yaml' 
else:
    version = f'yolo{v}{size}.pt'
configuration = '/home/u366836/thesis/CODE/TotalSegmentator/YOLO/yolo_tune_config.yaml'
model = YOLO(version)   

# Tuning settings:
search_space = {
        "lr0": (1e-5, 1e-1), # def = 0.01, using default search space range
        "lrf": (0.01, 1.0), # def = 0.01, using default search space range
        "box": (0.02, 7.5), # def = 7.5, default search space range is (0.02, 0.2)
        "cls": (0.1, 0.5), # def = 0.5, default search space range is (0.2, 4.0)
        "dfl": (0.4, 3), # def = 1.5, default search space range is (0.4, 6.0)
        "weight_decay": (0.0, 0.001), # def = 0.0005, using default search space range
        "momentum": (0.6, 0.98), # def = 0.937, using default search space range
        "warmup_epochs": (0.0, 5.0), # def = 3.0, using default search space range
        "warmup_momentum": (0.0, 0.95), # def = 0.8, using default search space range
    } 
epochs = 20

# Tune model
results = model.tune(data=configuration, split='val', epochs=epochs, iterations=100, name=f"yolov5_tune_{size}", resume=True)
