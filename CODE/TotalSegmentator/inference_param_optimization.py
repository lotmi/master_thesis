"""
Script to tune inference hyperparameters for RetinaNet

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

# Create grid
confs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
nms_threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
data_path = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL"
[val_loader] = create_dataloaders(device, data_path, sets=["val"], batch_size=4)

# Run grid search
best_conf = 0
best_iou = 0
best_weighted_f1 = 0
best_run = 0
run_counter = 1
for score_thresh in confs:
    for nms_thresh in nms_threshs:
        # Create model 
        model_path = "/home/u366836/thesis/CODE/TotalSegmentator/RetinaNet/RERUN_best_bone_retinanet.pth"
        model = BoneRetinaNet(num_classes=2, score_thresh=score_thresh, nms_thresh=nms_thresh, anchor_generator='small')
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
        model.eval()

        # Compute false/true positives/negatives for precision, recall and F1 metrics
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                for pred, tgt in zip(outputs, targets):
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
                    
        # Compute precision/recall/F1 
        if tp + fp + fn > 0:
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
        else:
            precision = recall_score = f1 = 0.0
        
        weighted_f1 = 0.25*precision + 0.75*recall
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            best_conf = score_thresh
            best_iou = nms_thresh
            best_run = run_counter
            print(f"\nSaved improved iou ({best_iou}) and conf ({best_conf}) at run {run_counter} for an F1 of {best_weighted_f1}", flush=True)
        run_counter += 1
        print(f"Results for conf={score_thresh} and iou={nms_thresh}:", flush=True)
        print("Precision:", precision, flush=True)
        print("Recall:", recall, flush=True)
        print("Weighted F1:", weighted_f1, flush=True)

print(f"Best iou={best_iou} and conf={best_conf} at run {best_run} for an F1 of {best_weighted_f1}", flush=True)