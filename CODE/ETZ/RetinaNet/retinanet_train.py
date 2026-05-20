"""
Script to train a RetinaNet model on the bone lesion detection task.

"""

# Import libraries
import torch
import pandas as pd
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.anchor_utils import AnchorGenerator
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet, EarlyStopping
from tqdm import tqdm

from retinanet_eval_loop import retinanet_evaluation


# Set up experiment
exp = "boneboxes"
data_path = "/wecare/home/lotte/Thesis/DATA/ETZ/PASCAL_SF_boneboxes"

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
batch_size = 1
[train_loader, val_loader, test_loader] = create_dataloaders(device, data_path, sets=["train", "val", "test"], batch_size=batch_size)
print(f"Created train ({len(train_loader)}), val ({len(val_loader)}) and test ({len(test_loader)}) dataloaders.", flush=True) # = len datasets / batch size

# Hyperparameters (as (pre-)tuned)
lr = 0.01
decay = 0.0001
factor = 0.2
anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5) # smaller sizes than default
gamma = 2

# Create model 
pretrained_weights = "/wecare/home/lotte/Thesis/CODE/ETZ/pretrained_weights/bone_retinanet.pth"
model = BoneRetinaNet(num_classes=2, anchor_generator=anchor_generator, gamma=gamma)
model.load_state_dict(torch.load(pretrained_weights, weights_only=True))
print(model, flush=True)
model.to(device)

# Define: Optimizer, LR scheduler, Mixed precision (scaler) (for faster learning), Early stopping, Number of epochs
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
fitness_scores = []

# Training loop
print('Start training RetinaNet', flush=True)
completed_epochs = 0
for epoch in tqdm(range(num_epochs)):
    print(f'Working on epoch {epoch}', flush=True)
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, (images, targets) in enumerate(train_loader): # go over all image batches in the train dataloader
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

    # Validation loop: compute validation loss
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            # Compute loss for the full val set
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)  # Store validation loss
    
    # Scheduler and early stop step
    lr_scheduler.step(val_loss)
    early_stopper.step(val_loss)

    # Validation loop: compute validation precision, recall, F1 and mAP
    model.eval()        
    precision, recall, f1, map50, map50_95 = retinanet_evaluation(model, val_loader) 
    precision_scores.append(precision) # Store precision score
    f1_scores.append(f1)  # Store F1 score
    recall_scores.append(recall)  # Store recall score
    map50_scores.append(map50) # Store map50 score
    map50_95_scores.append(map50_95) # Store map50-95 score

    # Save best model
    fitness = 0.9*map50 + 0.1*map50_95
    if fitness > best_fitness:
        best_fitness = fitness
        torch.save(model.state_dict(), f"/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_best_retinanet.pth")
        print("Best model saved!")
    fitness_scores.append(fitness)

    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"F1: {f1:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"mAP50: {map50:.4f} | "
        f"mAP50-95: {map50_95:.4f} | "
        f"Fitness: {fitness:.4f} | "
        f"Time: {time.time() - start_time:.2f}s", 
        flush=True)

    # Periodic save
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_retinanet_epoch_{epoch+1}.pth")

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
            "map50-95": map50_95_scores,
            "fitness": fitness_scores}
df = pd.DataFrame(metrics)
df.to_csv(f'/wecare/home/lotte/Thesis/CODE/ETZ/RetinaNet/runs/{exp}_retinanet_metrics.csv')

print("Completed script.")
