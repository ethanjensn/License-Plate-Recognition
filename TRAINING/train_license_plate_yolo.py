#!/usr/bin/env python3
"""
Train a custom YOLO model for license plate detection
"""

# python train_license_plate_yolo.py --dataset "plate-license-5_yolo" --model_size n --epochs 50 --batch_size 8

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def setup_dataset_structure(dataset_path, output_path="license_plate_dataset"):
    """
    Ensure dataset has proper YOLO structure:
    license_plate_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Map your folder names to YOLO names
    folder_mapping = {
        'train': 'train',
        'valid': 'val',  # YOLO uses 'val' not 'valid'
        'test': 'test'
    }
    
    for source_folder, target_folder in folder_mapping.items():
        source_path = dataset_path / source_folder
        
        if not source_path.exists():
            print(f"Warning: {source_folder} folder not found in {dataset_path}")
            continue
            
        print(f"Processing {source_folder} -> {target_folder}")
        
        # Copy images
        images_src = source_path / 'images'
        labels_src = source_path / 'labels'
        
        # Alternative structure: images and labels might be directly in train/valid/test
        if not images_src.exists():
            images_src = source_path
            labels_src = source_path
        
        images_dst = output_path / 'images' / target_folder
        labels_dst = output_path / 'labels' / target_folder
        
        # Copy image files
        for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_file in images_src.glob(img_ext):
                shutil.copy2(img_file, images_dst)
                
        # Copy label files
        for label_file in labels_src.glob('*.txt'):
            # Skip if it's not a YOLO label file (should have same name as image)
            img_name = label_file.stem
            if any((images_dst / f"{img_name}{ext}").exists() 
                   for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                shutil.copy2(label_file, labels_dst)
    
    return output_path

def create_dataset_yaml(dataset_path, class_names=['license_plate']):
    """Create dataset.yaml file for YOLO training"""
    dataset_path = Path(dataset_path)
    
    config = {
        'path': str(dataset_path.absolute()),  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/val',      # val images (relative to 'path')
        'test': 'images/test',    # test images (optional)
        
        # Classes
        'nc': len(class_names),   # number of classes
        'names': {i: name for i, name in enumerate(class_names)}  # class names
    }
    
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config: {yaml_path}")
    return yaml_path

def train_yolo_model(dataset_yaml, 
                    model_size='n',  # n, s, m, l, x
                    epochs=100,
                    imgsz=640,
                    batch_size=16,
                    project='license_plate_detection',
                    name='yolo_lpr'):
    """Train YOLO model"""
    
    # Load a model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)  # load a pretrained model (recommended for training)
    
    print(f"Training {model_name} on {dataset_yaml}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
    
    # Train the model
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project,
        name=name,
        
        # Training hyperparameters
        lr0=0.01,          # initial learning rate
        momentum=0.937,     # SGD momentum/Adam beta1
        weight_decay=0.0005, # optimizer weight decay 5e-4
        warmup_epochs=3,    # warmup epochs (fractions ok)
        warmup_momentum=0.8, # warmup initial momentum
        warmup_bias_lr=0.1, # warmup initial bias lr
        
        # Data augmentation
        hsv_h=0.015,       # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,         # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,         # image HSV-Value augmentation (fraction)
        degrees=0.0,       # image rotation (+/- deg)
        translate=0.1,     # image translation (+/- fraction)
        scale=0.5,         # image scale (+/- gain)
        shear=0.0,         # image shear (+/- deg)
        perspective=0.0,   # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,        # image flip up-down (probability)
        fliplr=0.5,        # image flip left-right (probability)
        mosaic=1.0,        # image mosaic (probability)
        mixup=0.0,         # image mixup (probability)
        
        # Other settings
        save_period=10,     # Save checkpoint every x epochs
        cache=False,        # True/ram, disk or False
        device='',          # device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,          # number of worker threads for data loading
        exist_ok=True,      # existing project/name ok, do not increment
        pretrained=True,    # use pretrained model
        optimizer='SGD',    # optimizer (SGD, Adam, AdamW, RMSProp)
        verbose=True,       # verbose output
        seed=0,            # random seed for reproducibility
        deterministic=True, # whether to enable deterministic mode
        single_cls=False,   # treat as single-class dataset
        rect=False,         # rectangular training
        cos_lr=False,       # cosine learning rate scheduler
        close_mosaic=10,    # disable mosaic augmentation for final epochs
        resume=False,       # resume training from last checkpoint
    )
    
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")
    
    return model, results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO for License Plate Detection')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset folder with train/valid/test subdirs')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--project', type=str, default='license_plate_detection',
                       help='Project name for saving results')
    parser.add_argument('--name', type=str, default='yolo_lpr',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    print("=== YOLO License Plate Detection Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: YOLOv8{args.model_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    
    # Check if dataset is already in YOLO format
    dataset_path = Path(args.dataset)
    dataset_yaml_path = dataset_path / 'dataset.yaml'
    
    if dataset_yaml_path.exists():
        # Dataset is already in YOLO format, use it directly
        print("\n1. Using existing YOLO dataset...")
        processed_dataset = dataset_path
        dataset_yaml = dataset_yaml_path
    else:
        # Step 1: Setup dataset structure
        print("\n1. Setting up dataset structure...")
        processed_dataset = setup_dataset_structure(args.dataset)
        
        # Step 2: Create dataset YAML
        print("\n2. Creating dataset configuration...")
        dataset_yaml = create_dataset_yaml(processed_dataset, class_names=['license_plate'])
    
    # Step 3: Train model
    print("\n3. Starting training...")
    model, results = train_yolo_model(
        dataset_yaml=dataset_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        project=args.project,
        name=args.name
    )
    
    print("\n=== Training Complete ===")
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    print(f"Best model saved to: {best_model_path}")
    print(f"To use this model, update DET_MODEL in run_crnn_lpr.py to: '{best_model_path}'")

if __name__ == "__main__":
    main() 