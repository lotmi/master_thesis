"""
Script to train a RetinaNet model on the bone detection task.

"""

# Import libraries
import torch
import os
import pandas as pd
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet, EarlyStopping


# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
data_path = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL"
batch_size = 4
[train_loader, val_loader] = create_dataloaders(device, data_path, sets=["train", "val"], batch_size=batch_size)
print(f"Created train ({len(train_loader)}) and val ({len(val_loader)}) dataloaders.", flush=True) # = len datasets / batch size

# Hyperparameters (as tuned)
tuning_results = pd.read_csv('/home/u366836/thesis/CODE/TotalSegmentator/RetinaNet/retinanet_tuning_results.csv')
hyperparams = tuning_results[tuning_results['val_loss']==tuning_results['val_loss'].min()].reset_index()
print("Using tuned hyperparameters:", hyperparams, flush=True)
lr = hyperparams['lr'][0]
decay = hyperparams['decay'][0]
factor = hyperparams['factor'][0]
if hyperparams['anchors'][0] == "small":
    anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5) # smaller sizes than default
elif hyperparams['anchors'][0] == "default":
    anchor_generator = "DEFAULT"
else:
    print("No anchor setting found.", flush=True)
    
# Create model 
score_thresh=0.25 # confidence threshold for inference
model = BoneRetinaNet(num_classes=2, anchor_generator=anchor_generator)
print(model, flush=True)
model.to(device)

# Define: Optimizer, LR scheduler, Mixed precision (scaler), Early stopping, Number of epochs
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, weight_decay=decay)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=factor, min_lr=1e-8) 
scaler = torch.amp.GradScaler()
early_stopper = EarlyStopping(patience=25, min_delta=0.001)
num_epochs = 200
best_val_loss = float("inf")
best_fitness = 0

# Create placeholders for metrics across epochs
train_losses = []
val_losses = []
f1_scores = []
precision_scores = []
recall_scores = []
map50_scores = []
map50_95_scores = []

# Training loop
print('Start training RetinaNet', flush=True)
completed_epochs = 0
for epoch in range(num_epochs):
    print(f'Working on epoch {epoch}', flush=True)
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += losses.item()
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)  # Store training loss

    # Validation loop
    model.eval()
    val_loss = 0.0
    tp, fp, fn = 0, 0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            # Compute loss for the full val set
            model.train()
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            
            # Compute false/true positives/negatives for precision, recall and F1 metrics
            model.eval()
            outputs = model(images)
            for pred, tgt in zip(outputs, targets):
                conf_mask = pred["scores"] >= score_thresh
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

    val_loss /= len(val_loader)
    val_losses.append(val_loss)  # Store validation loss

    # Compute precision/recall/F1/mAP if available
    if tp + fp + fn > 0:
        precision = tp / (tp + fp + 1e-6)
        recall_score = tp / (tp + fn + 1e-6)
        f1_score = 2 * precision * recall_score / (precision + recall_score + 1e-6)
    else:
        precision = recall_score = f1_score = 0.0
        
    map_metric = MeanAveragePrecision()
    map_metric.update(all_preds, all_targets)
    map_results = map_metric.compute()
    map50 = map_results['map_50'].item()
    map50_95 = map_results['map'].item()
    
    precision_scores.append(precision)
    f1_scores.append(f1_score)  # Store F1 score
    recall_scores.append(recall_score)  # Store recall score
    map50_scores.append(map50)
    map50_95_scores.append(map50_95)

    # Save best model
    fitness = 0.9*map50 + 0.1*map50_95
    if fitness > best_fitness:
        best_fitness = fitness
        torch.save(model.state_dict(), "best_bone_retinanet.pth")
        print("Best model saved!")

    # Scheduler and early stop step
    lr_scheduler.step(val_loss)
    early_stopper.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"F1: {f1_score:.4f} | "
          f"Precision: {precision:.4f} | "
          f"Recall: {recall_score:.4f} | "
          f"mAP50: {map50:.4f} | "
          f"mAP50-95: {map50_95:.4f} | "
          f"Time: {time.time() - start_time:.2f}s", 
          flush=True)

    # Periodic save
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"bone_retinanet_epoch_{epoch+1}.pth")

    completed_epochs += 1
    
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
        break


# Save metrics to .csv
metrics = {"epoch": list(range(completed_epochs)), 
           "train_loss": train_losses, 
           "val_loss": val_losses, 
           "f1": f1_scores, 
           "precision": precision_scores,
           "recall": recall_scores,
           "map50": map50_scores,
           "map50-95":map50_95_scores }
df = pd.DataFrame(metrics)
df.to_csv('bone_retinanet_metrics.csv')