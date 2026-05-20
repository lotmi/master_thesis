"""
Script to tune the hyperparameters for model evaluation/inference:
- Confidence threshold (def. = 0.25)
- IoU for NMS (def. = 0.5)

Metric to maximize: weighted F1 (trading off both precision (0.25) and recall (0.75))

"""

# Import packages
import torch
from tqdm import tqdm
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet
from retinanet_eval_loop import retinanet_evaluation

def compute_weighted_f1(precision, recall, pw=0.25, rw=0.75):
       if pw + rw != 1.0:
              print("Weights for precision and recall do not add up to 1. Returning 0 for weighted F1 and continuing...")
              return 0
       else:
              weighted_f1 = pw*precision + rw*recall
              return weighted_f1

# Set these variables yourself to indicate for which model the best-performing inference parameters should be found
exp = "SF"
model_path = f"/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_best_retinanet.pth"

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
batch_size = 4
data_path = "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_Sample"
[val_loader] = create_dataloaders(device, data_path, sets=["val"], batch_size=batch_size)

# Parameter Grid and Grid Search Loop
confs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ious = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

best_conf = 0
best_iou = 0
best_weighted_f1 = 0
for conf in tqdm(confs):
       for iou in ious:
              # Create model 
              model = BoneRetinaNet(num_classes=2, score_thresh=conf, nms_thresh=iou, anchor_generator='small')
              model.load_state_dict(torch.load(model_path, weights_only=True))
              model.to(device)
              model.eval()
              # Inference with given parameters
              precision, recall, f1, map50, map50_95 = retinanet_evaluation(model, val_loader, conf_thres=conf, iou_thres=iou)
              weighted_f1 = compute_weighted_f1(precision, recall)
              if weighted_f1 > best_weighted_f1:
                     best_weighted_f1 = weighted_f1
                     best_conf = conf
                     best_iou = iou
                     print(f"Saved improved inference parameters at a weighted F1 score of {best_weighted_f1}: conf = {best_conf} and iou = {best_iou}")
print(f"Best inference parameters for RetinaNet were found to be: conf = {best_conf} and iou = {best_iou}, giving a F1 score of {best_weighted_f1}.")
