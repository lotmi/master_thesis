"""
Definitions of the RetinaNet dataset class and dataloaders.

"""

# Import libraries
import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

""" Define RetinaNet dataset class """
class RetinaNetDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f) # annotations is now a dictionary with "image_name" : [[bounding box]] entries
        self.data = []
        for image_name, boxes in self.annotations.items():
            self.data.append((image_name, boxes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, boxes = self.data[idx]
        filename = os.path.basename(filename)
        img_path = os.path.join(self.image_dir, filename)
        
        image = Image.open(img_path).convert("RGB")
        img_arr = np.array(image)
        # Pre-processing: fix image dimensions (permute) and normalize 0-255 range to 0-1
        image = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0 
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)  # One class (bones)
        target = {'boxes': boxes_tensor, 'labels': labels}
        
        return image, target
    
""" Define RetinaNet dataloaders """
def create_dataloaders(device, data_path, sets=["train", "val", "test"], batch_size=1):
    
    def collate_fn(batch): # Custom collate function to handle batching of images (not necessarily same size) and targets 
        images, targets = zip(*batch)
        images = [image.to(device) for image in images]  
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 
        return images, targets
    
    # Create datasets and dataloaders
    loaders = []
    if "train" in sets:
        train = RetinaNetDataset(
            image_dir=os.path.join(data_path, 'train/images'),
            annotation_file=os.path.join(data_path, 'train/annotations.json'))
        train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        loaders.append(train_loader)
        
    if "val" in sets:
        val = RetinaNetDataset(
            image_dir=os.path.join(data_path, 'val/images'),
            annotation_file=os.path.join(data_path, 'val/annotations.json'))
        val_loader   = DataLoader(val,   batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        loaders.append(val_loader)
        
    if "test" in sets:
        test = RetinaNetDataset(
            image_dir=os.path.join(data_path, 'test/images'),
            annotation_file=os.path.join(data_path, 'test/annotations.json'))
        test_loader  = DataLoader(test,  batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        loaders.append(test_loader)
    
    return loaders