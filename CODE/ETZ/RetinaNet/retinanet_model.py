"""
Class definitions for:
* The 'BoneRetinaNet' used for this thesis project
* RetinaNet's binary classification head (lesion vs. background)
* Early stopping mechanism for RetinaNet training

"""

# Import the relevant libraries
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead   

""" Define custom RetinaNet head (for binary classification) """
class BinaryHead(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes, prior_probability=0.01) # Original values
        
        
""" RetinaNet model for bone lesion detection """           
class BoneRetinaNet(nn.Module):
    """
    Custom RetinaNet for lesion detection:
    Class 0 = background, Class 1 = lesion
    """
    def __init__(self, num_classes=2, score_thresh=0.05, nms_thresh=0.5, anchor_generator="DEFAULT", weights="DEFAULT", alpha=0.25, gamma=2): 
        super(BoneRetinaNet, self).__init__()

        # Load backbone 
        backbone_model = retinanet_resnet50_fpn_v2(weights=weights)
        backbone = backbone_model.backbone

        # Create RetinaNet with custom head (for binary classification)
        if anchor_generator=="DEFAULT":
            anchor_generator = AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)), 
                aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        elif anchor_generator=='small': # Might be more suitable for small bones / lesions
            anchor_generator = AnchorGenerator(
                sizes=((16,), (32,), (64,), (128,), (256,)), 
                aspect_ratios=((0.5, 1.0, 2.0),) * 5) # smaller sizes than default

        self.retinanet = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh, 
            anchor_generator=anchor_generator,
            head=BinaryHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)) # Use the modified head
        self.alpha = alpha
        self.gamma = gamma
        
    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs["cls_logits"]
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    inputs=cls_logits_per_image[valid_idxs_per_image],
                    targets=gt_classes_target[valid_idxs_per_image],
                    alpha=self.alpha,
                    gamma=self.gamma,
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)
            
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

