"""
Class definitions for:
* The 'BoneRetinaNet' used for this thesis project
* RetinaNet's binary classification head (bone vs. background)
* Early stopping mechanism for RetinaNet training

"""

# Import the relevant libraries
import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead   


""" Define custom RetinaNet head (for binary classification) """
class BinaryHead(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes, prior_probability=0.01) # Original values
        
        
""" RetinaNet model for bone detection """           
class BoneRetinaNet(nn.Module):
    """
    Custom RetinaNet for bone detection:
    Class 0 = background, Class 1 = bone
    """
    def __init__(self, num_classes=2, score_thresh=0.05, nms_thresh=0.5, anchor_generator="DEFAULT", weights="DEFAULT"):
        super(BoneRetinaNet, self).__init__()

        # Load backbone 
        backbone_model = retinanet_resnet50_fpn_v2(weights=weights)
        backbone = backbone_model.backbone

        # Create RetinaNet with custom head (for binary classification)
        if anchor_generator=="DEFAULT":
            anchor_generator = AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)), 
                aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        elif anchor_generator=="small":
            anchor_generator = AnchorGenerator(
                sizes=((16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5) # smaller sizes than default

        self.retinanet = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            anchor_generator=anchor_generator,
            head=BinaryHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)) # Use the modified head
            
    def forward(self, images, targets=None):
        return self.retinanet(images, targets)
    
    
    
""" Early stopping function during RetinaNet training """    
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
