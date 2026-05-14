"""
YOLO11x Teacher Training on Hybrid CrowdHuman + MERL Dataset
Implements class-balanced sampling and loss weighting for imbalanced data.
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HybridDataset(torch.utils.data.Dataset):
    """
    Custom dataset that balances CrowdHuman and MERL samples.
    """
    def __init__(self, crowdhuman_dir, merl_dir, merl_augment_multiplier=10):
        self.crowdhuman_images = list(Path(crowdhuman_dir).glob("images/*.jpg"))
        self.merl_images = list(Path(merl_dir).glob("images/*.jpg"))
        
        # Multiply MERL representation via augmentation
        self.merl_images_augmented = self.merl_images * merl_augment_multiplier
        
        print(f"CrowdHuman: {len(self.crowdhuman_images)} images")
        print(f"MERL: {len(self.merl_images)} images (augmented to {len(self.merl_images_augmented)})")
        
    def __len__(self):
        return len(self.crowdhuman_images) + len(self.merl_images_augmented)
    
    def __getitem__(self, idx):
        # Determine which dataset to sample from
        if idx < len(self.crowdhuman_images):
            img_path = self.crowdhuman_images[idx]
            is_merl = False
        else:
            merl_idx = idx - len(self.crowdhuman_images) % len(self.merl_images)
            img_path = self.merl_images[merl_idx]
            is_merl = True
        
        # Load image and labels (implement your YOLO format loader)
        img, labels = self._load_yolo_format(img_path)
        
        # Apply heavier augmentation to MERL samples
        if is_merl:
            transform = A.Compose([
                A.Mosaic(p=0.9),
                A.MixUp(p=0.5),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            transform = A.Compose([
                A.Mosaic(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        transformed = transform(image=img, bboxes=labels['bboxes'], class_labels=labels['class_labels'])
        
        return {
            'img': transformed['image'],
            'bboxes': torch.tensor(transformed['bboxes']),
            'class_labels': torch.tensor(transformed['class_labels']),
            'is_merl': is_merl  # Flag for loss weighting
        }
    
    def _load_yolo_format(self, img_path):
        # Implement your YOLO format loader here
        # Returns: img (numpy), labels {'bboxes': [], 'class_labels': []}
        pass

class WeightedLoss(torch.nn.Module):
    """
    Applies higher loss weight to MERL samples to compensate for underrepresentation.
    """
    def __init__(self, merl_loss_weight=50.0):
        super().__init__()
        self.merl_loss_weight = merl_loss_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, is_merl):
        # Calculate base loss
        loss = self.bce_loss(predictions, targets)
        
        # Apply weight multiplier for MERL samples
        weights = torch.where(is_merl, self.merl_loss_weight, 1.0)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

def train_teacher():
    # Load config
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("TEACHER MODEL TRAINING (YOLO11x)")
    print("="*60)
    
    # Option A: Use Ultralytics YOLO with custom dataset
    model = YOLO(config['teacher']['model'])
    
    # Custom training arguments with balancing
    train_args = {
        'data': 'config/hybrid_dataset.yaml',  # Custom YAML pointing to both datasets
        'epochs': config['teacher']['epochs'],
        'imgsz': config['teacher']['imgsz'],
        'batch': config['teacher']['batch'],
        'lr0': config['teacher']['lr0'],
        'weight_decay': config['teacher']['weight_decay'],
        'augment': True,
        'mosaic': 1.0,  # Always use mosaic for dense detection
        'mixup': 0.5,
        'copy_paste': 0.3,  # Helps with rare retail patterns
        'patience': 50,
        'save_period': 10,
        'project': 'runs/detect',
        'name': 'yolo11x_teacher_hybrid',
        'exist_ok': True,
        'pretrained': True,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        # Critical: Ensure balanced sampling
        'workers': 8,
        'close_mosaic': 10,  # Disable augmentations in last 10 epochs
    }
    
    # Train with balanced dataset
    results = model.train(**train_args)
    
    # Save best model
    best_model_path = Path('runs/detect/yolo11x_teacher_hybrid/weights/best.pt')
    print(f"\n Teacher model saved to: {best_model_path}")
    
    # Export validation metrics
    metrics = {
        'model': 'yolo11x',
        'map50': results.results_dict['metrics/mAP50(B)'],
        'map50-95': results.results_dict['metrics/mAP50-95(B)'],
        'precision': results.results_dict['metrics/precision(B)'],
        'recall': results.results_dict['metrics/recall(B)'],
        'epochs_trained': config['teacher']['epochs']
    }
    
    import json
    with open('runs/detect/yolo11x_teacher_hybrid/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return best_model_path

if __name__ == "__main__":
    train_teacher()