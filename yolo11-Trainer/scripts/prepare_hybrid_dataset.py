"""
Prepares balanced hybrid dataset by creating symlinks and metadata.
Ensures MERL samples are properly weighted during training.
"""

import os
import shutil
from pathlib import Path
import random

def prepare_hybrid_dataset():
    crowdhuman_dir = Path("/datasets/crowdhuman")
    merl_dir = Path("/datasets/merl")
    output_dir = Path("/datasets/hybrid_combined")
    
    # Create directory structure
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Copy CrowdHuman (sample to balance)
    print("Processing CrowdHuman...")
    ch_images = list((crowdhuman_dir / "images").glob("*.jpg"))
    random.seed(42)
    ch_selected = random.sample(ch_images, 8000)  # Downsample to balance
    
    for i, img_path in enumerate(ch_selected):
        shutil.copy(img_path, output_dir / "images" / "train" / f"ch_{i:05d}.jpg")
        shutil.copy(img_path.with_suffix(".txt"), 
                   output_dir / "labels" / "train" / f"ch_{i:05d}.txt")
    
    # Copy MERL (repeat for balancing)
    print("Processing MERL...")
    merl_images = list((merl_dir / "images").glob("*.jpg"))
    
    # Repeat MERL 10x to balance with CrowdHuman
    for repeat in range(10):
        for i, img_path in enumerate(merl_images):
            idx = i + (repeat * len(merl_images))
            shutil.copy(img_path, output_dir / "images" / "train" / f"merl_{repeat}_{i:05d}.jpg")
            shutil.copy(img_path.with_suffix(".txt"), 
                       output_dir / "labels" / "train" / f"merl_{repeat}_{i:05d}.txt")
    
    print(f" Hybrid dataset prepared: {output_dir}")
    print(f"   CrowdHuman: {len(ch_selected)} images")
    print(f"   MERL: {len(merl_images) * 10} images (10x repeated)")

if __name__ == "__main__":
    prepare_hybrid_dataset()