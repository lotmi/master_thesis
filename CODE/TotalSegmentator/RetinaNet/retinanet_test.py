"""
Script to evaluate a RetinaNet model on the lesion detection task with conventional inference.

Computing: Precision, Recall, F1, mAP.

"""

# Import libraries
import torch
import os
import pandas as pd
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import box_iou
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet, EarlyStopping
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Provide optimized evaluation parameters:
score_thresh=0.3
nms_thresh=0.45


# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
data_path = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL"
[test_loader] = create_dataloaders(device, data_path, sets=["test"], batch_size=1)

# Create model 
model_path = "/home/u366836/thesis/CODE/TotalSegmentator/RetinaNet/RERUN_best_bone_retinanet.pth"
model = BoneRetinaNet(num_classes=2, score_thresh=score_thresh, nms_thresh=nms_thresh, anchor_generator='small')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()

# Compute false/true positives/negatives for precision, recall and F1 metrics
all_preds = []
all_targets = []
tp, fp, fn = 0, 0, 0
with torch.no_grad():
    for images, targets in test_loader:
        outputs = model(images)
        for pred, tgt in zip(outputs, targets):
            all_preds.append(pred)
            all_targets.append(tgt)
            pred_boxes = pred["boxes"]
            tgt_boxes = tgt["boxes"]
            if pred_boxes.size(0) == 0:
                fn += tgt_boxes.size(0)
                continue
            if tgt_boxes.size(0) == 0:
                fp += pred_boxes.size(0)
                continue
            ious = box_iou(pred_boxes, tgt_boxes)
            matched_gt = torch.zeros(tgt_boxes.size(0), dtype=torch.bool, device=device)
            for pred_iou in ious:
                max_iou, gt_idx = pred_iou.max(0)
                if max_iou >= 0.5 and not matched_gt[gt_idx]:
                    tp += 1
                    matched_gt[gt_idx] = True
                else:
                    fp += 1
            fn += (~matched_gt).sum().item()
            
# Compute precision/recall/F1/mAP
if tp + fp + fn > 0:
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
else:
    precision = recall_score = f1 = 0.0
    
map_metric = MeanAveragePrecision()
map_metric.update(all_preds, all_targets)
map_results = map_metric.compute()
map50 = map_results['map_50'].item()
map50_95 = map_results['map'].item()

print(f"\nResults for conf. threshold {score_thresh}:", flush=True)
print("Precision:", precision, flush=True)
print("Recall:", recall, flush=True)
print("F1:", f1, flush=True)
print("mAP50:", map50, flush=True)
print("mAP50-95:", map50_95, flush=True)