"""
Script to tune the training hyperparameters for RetinaNet:
* LR
* Learning rate factor
* Weight decay
* Anchor size

"""

# Import libraries
import torch
import pandas as pd
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.anchor_utils import AnchorGenerator
from retinanet_dataprep import create_dataloaders
from retinanet_model import BoneRetinaNet

# Connect to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Create datasets and dataloaders
data_path = "/home/u366836/thesis/DATA/TotalSegmentator/PASCAL_sample"
batch_size = 1
[train_loader, val_loader] = create_dataloaders(device, data_path, sets=["train", "val"], batch_size=batch_size)
print(f"Created train ({len(train_loader)}) and val ({len(val_loader)}) dataloaders.", flush=True) # = len datasets / batch size

# Hyperparameter search space
file_name = "retinanet_tuning_results.csv"
lrs = [1e-2, 1e-3, 1e-4,] # default = 1e-3
decays = [1e-2, 1e-4, 0] # default = 0
factors = [0.5, 0.2, 0.1] # default = 0.1
anchor_sizes = ["default", "small"]
anchor_generators = [AnchorGenerator( # default
                        sizes=((32,), (64,), (128,), (256,), (512,)), 
                        aspect_ratios=((0.5, 1.0, 2.0),) * 5),
                    AnchorGenerator( # smaller sizes than default
                        sizes=((16,), (32,), (64,), (128,), (256,)),  
                        aspect_ratios=((0.5, 1.0, 2.0),) * 5)]

results = {"lr": [], 
            "decay": [], 
            "factor": [], 
            "anchors": [], 
            "val_loss": []}

# Run grid search loop
for lr in lrs:
    for decay in decays:
        for factor in factors:
            for anchor_size, anchor_generator in zip(anchor_sizes, anchor_generators):
                # Check if this configuration was already trained
                configuration = {"lr":lr, "decay":decay, "factor":factor, "anchors":anchor_size}
                
                # Create model 
                model = BoneRetinaNet(num_classes=2, anchor_generator=anchor_generator)
                model.to(device)
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(params, lr = lr, weight_decay=decay) 
                lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=factor, min_lr=1e-8)
                scaler = torch.amp.GradScaler()
                num_epochs = 20
                
                # Train and evaluate configuration 
                iteration_count += 1
                print(f"\nTraining for configuration {iteration_count}/54: {configuration}", flush=True)
                best_val_loss = float("inf")
                for epoch in range(num_epochs): # Run epochs
                    model.train()
                    start_time = time.time()
                    for i, (images, targets) in enumerate(train_loader): # Train model
                        optimizer.zero_grad()
                        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            loss_dict = model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                        scaler.scale(losses).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    val_loss = 0.0 # Compute validation loss
                    with torch.no_grad():
                        for images, targets in val_loader:
                            model.train()
                            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                loss_dict = model(images, targets)
                                losses = sum(loss for loss in loss_dict.values())
                            val_loss += losses.item()
                    val_loss /= len(val_loader)
                    if val_loss < best_val_loss: 
                        best_val_loss = val_loss
                print(f"Completed training. Best validation loss = {best_val_loss}. Saving results to .csv file.", flush=True)
                
                # Save configuration results
                results["lr"].append(lr) 
                results["decay"].append(decay)
                results["factor"].append(factor)
                results["anchors"].append(anchor_size)
                results["val_loss"].append(best_val_loss)
                results_df = pd.DataFrame({i:results[i] for i in results if i!="Unnamed: 0"}) # Save results to .csv
                results_df.to_csv(file_name)

