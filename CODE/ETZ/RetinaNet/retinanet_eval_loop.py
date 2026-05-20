"""
General RetinaNet evaluation loop for conventional inference.

The function computes: Recall, Precision, F1 and mAP scores.

"""

# Import libraries
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd

def retinanet_evaluation(model, val_loader, conf_thres=0.25, iou_thres=0.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    all_preds = []
    all_targets = []
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            # Compute false/true positives/negatives for precision, recall and F1 metrics
            outputs = model(images) 
            for pred, tgt in zip(outputs, targets):
                conf_mask = pred["scores"] >= conf_thres 
                pred = {'boxes':pred['boxes'][conf_mask], 'scores':pred['scores'][conf_mask], 'labels':pred['labels'][conf_mask]}
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

    # Compute precision/recall/F1/mAP metrics
    if tp + fp + fn > 0:
        precision = tp / (tp + fp + 1e-6)
        recall_score = tp / (tp + fn + 1e-6)
        f1_score = 2 * precision * recall_score / (precision + recall_score + 1e-6)
    else:
        precision = recall_score = f1_score = 0.0
    map_metric = MeanAveragePrecision() # iou_thresholds=None equals the default 0.5-0.95 (0.05 steps) range
    map_metric.update(all_preds, all_targets)
    map_results = map_metric.compute()
    map50 = map_results['map_50'].item()
    map50_95 = map_results['map'].item()

    return precision, recall_score, f1_score, map50, map50_95

