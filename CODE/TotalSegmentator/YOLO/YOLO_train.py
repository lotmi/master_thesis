"""
Script to train a YOLO Model (either version 5 or 26).

"""

# Import packages
from ultralytics import YOLO

# Set these yourself to determine which model version/edition is trained:
v = "5" # 5 or 26
size = "s"

hyperparameter_dict = {'yolov5n.yaml':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov5_tune_nano/best_hyperparameters.yaml',
                       'yolov5s.yaml':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov5_tune_small/best_hyperparameters.yaml',
                       'yolov5m.yaml':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov5_tune_medium/best_hyperparameters.yaml',
                       'yolov5l.yaml':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov5_tune_large/best_hyperparameters.yaml',
                       'yolo26n.pt':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov26_tune_nano/best_hyperparameters.yaml',
                       'yolo26s.pt':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov26_tune_small/best_hyperparameters.yaml',
                       'yolo26m.pt':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov26_tune_medium/best_hyperparameters.yaml',
                       'yolo26l.pt':'/home/u366836/thesis/CODE/TotalSegmentator/YOLO/tuning_runs/detect/yolov26_tune_large/best_hyperparameters.yaml'
} # tuned hyperparameters per YOLO version/size

# Load, train and evaluate the model
if v == "5":
    version = f'yolov{v}{size}.yaml' 
else:
    version = f'yolo{v}{size}.pt'
vn = version[0:7] # version number
hyperparams = hyperparameter_dict[version]
configuration = '/home/u366836/thesis/CODE/TotalSegmentator/YOLO/yolo_config.yaml'
model = YOLO(version)
train_results = model.train(data=configuration, cfg=hyperparams, epochs=200, patience=25, split='train', name=f'{vn}_train')
test_results = model.val(data=configuration, split='test', name=f'{vn}_test', visualize=True)
