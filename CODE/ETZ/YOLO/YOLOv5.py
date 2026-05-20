"""
Script to run a YOLOv5 Model.

Code blocks can be commented out to run the training script for a specific data configuration (base, SF, ROI-SF).

"""

# Import packages
from ultralytics import YOLO
import subprocess
import glob


# Define YOLOv5 settings
size = 's'
pretrained_weights = f"/wecare/home/lotte/Thesis/CODE/ETZ/pretrained_weights/yolo5{size}.pt"
pretuned_hyperparams = f"/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/hyperparameters/YOLOv5{size}.yaml"

# # Load, train and evaluate the model (ON ORIGINAL DATA)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_config.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, epochs=200, patience=25, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'base_{size}_train') # patience=25
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'base_{size}_test', visualize=True)

# # Load, train and evaluate the model (BASELINES ON SAMPLED DATA)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_sampled_config.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, batch=1, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'base_{size}_train_sampled') # patience=25
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'base_{size}_test_sampled', visualize=True)

# # Load, train and evaluate the model (ON SF DATA WITH BONE BOXES)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_SF_boneboxes_config.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, batch=1, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'SF_boneboxes_{size}_train') # patience=25
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'SF_boneboxes_{size}_test', visualize=True)

# # Load, train and evaluate the model (ON SF DATA)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_SF.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'SF_{size}_train') # patience=25, batch=1
# for path in glob.glob("/tmp/pymp*"):
#                     subprocess.run(['rm', '-r', path], capture_output=True)
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'SF_{size}_test', visualize=True)

# NS dataset: Load, train and evaluate the model (ON ROISF DATA)
configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_config_NS_ROISF.yaml'
model = YOLO(pretrained_weights)
train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'NS_ROISF_train') # patience=25, batch=1
for path in glob.glob("/tmp/pymp*"):
                    subprocess.run(['rm', '-r', path], capture_output=True)
test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'NS_ROISF_test', visualize=True)

# # Secondary dataset: Load, train and evaluate the model (ON SF DATA)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/cross_yolo_config_SF.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'cross_SF_train') # patience=25, batch=1
# for path in glob.glob("/tmp/pymp*"):
#                     subprocess.run(['rm', '-r', path], capture_output=True)
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'cross_SF_test', visualize=True)

# # Secondary dataset: Load, train and evaluate the model (ON ROISF DATA)
# configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/cross_yolo_config_ROISF.yaml'
# model = YOLO(pretrained_weights)
# train_results = model.train(data=configuration, cfg=pretuned_hyperparams, epochs=200, patience=25, split='train', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'cross_ROISF_train') # patience=25, batch=1
# for path in glob.glob("/tmp/pymp*"):
#                     subprocess.run(['rm', '-r', path], capture_output=True)
# test_results = model.val(data=configuration, split='test', project=f'/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv5', name=f'cross_ROISF_test', visualize=True)

# After each YOLO run: clean pymp files from the temporary folder!
for path in glob.glob("/tmp/pymp*"):
                    subprocess.run(['rm', '-r', path], capture_output=True)
