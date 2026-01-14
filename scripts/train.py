#!/usr/bin/env python3
# Copyright (c) Remote Sensing Segmentation Training
# Unified training script for all datasets - Quick 1 epoch training

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import datasets
from datasets.loveda_dataset import LoveDADataset
from datasets.isaid_dataset import iSAIDDataset
from datasets.vaihingen_dataset import VaihingenDataset
from datasets.potsdam_dataset import PotsdamDataset
from datasets.landcoverai_dataset import LandCoveraiDataset
from datasets.openearthmap_dataset import OpenEarthMapDataset

# Import model
from models.segmentation_model import DINOv3SegmentationModel


# Dataset configurations - Full training without strict limits
# Very high epoch limit, only early stopping will end training
DATASET_CONFIGS = {
    'loveda': {
        'class': LoveDADataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'LoveDA',
        'train_split': 'Train',
        'val_split': 'Val',
        'num_classes': 7,
        'epochs': 10000  # No limit - train until early stopping
    },
    'isaid': {
        'class': iSAIDDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'iSAID',
        'train_split': 'train',
        'val_split': 'val',
        'num_classes': 16,
        'epochs': 10000  # No limit - train until early stopping
    },
    'vaihingen': {
        'class': VaihingenDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'Vaihingen',
        'train_split': 'train',
        'val_split': 'test',
        'num_classes': 5,
        'epochs': 10000  # No limit - train until early stopping
    },
    'potsdam': {
        'class': PotsdamDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'Potsdam',
        'train_split': 'train',
        'val_split': 'test',
        'num_classes': 6,
        'epochs': 10000  # No limit - train until early stopping
    },
    'landcoverai': {
        'class': LandCoveraiDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'LandCoverai',
        'train_split': 'train',
        'val_split': 'val',
        'num_classes': 5,
        'epochs': 10000  # No limit - train until early stopping
    },
    'openearthmap': {
        'class': OpenEarthMapDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'OpenEarthMap',
        'train_split': 'Val',
        'val_split': 'Val',
        'num_classes': 8,
        'epochs': 10000  # No limit - train until early stopping
    }
}


def load_backbone(backbone_name, pretrained_weights):
    """Load DINOv3 backbone with pretrained weights"""
    print(f"Loading backbone: {backbone_name}")

    # Import dinov3 models
    import dinov3.hub.backbones as backbones

    # Get backbone function
    backbone_fn = getattr(backbones, backbone_name)

    # Load backbone
    backbone = backbone_fn(pretrained=False)

    # Load pretrained weights if provided
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"Loading pretrained weights from: {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location='cpu')

        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove teacher prefix if exists
        state_dict = {k.replace('teacher.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

        # Load weights
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights")
    else:
        print("No pretrained weights loaded")

    return backbone


def calculate_metrics(preds, targets, num_classes, ignore_index=255):
    """Calculate segmentation metrics"""
    # Flatten
    preds = preds.flatten()
    targets = targets.flatten()

    # Remove ignore index
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]

    # Calculate metrics per class
    ious = []

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(float('nan'))

    # Calculate mean metrics
    ious = np.array(ious)
    miou = float(np.nanmean(ious))

    # Overall accuracy
    accuracy = (preds == targets).sum().item() / len(targets)

    return {
        'miou': miou,
        'accuracy': accuracy,
        'per_class_iou': [float(x) if not np.isnan(x) else None for x in ious]
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, num_classes, device):
    """Evaluate model on validation set"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Evaluating')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Forward
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs, dim=1)

        # Collect for metrics
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics, all_preds, all_targets


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, desc='Training')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Forward
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    return {'loss': avg_loss}


def train_dataset(dataset_name, args):
    """Train on a specific dataset"""
    print(f"\n{'='*80}")
    print(f"Training on {dataset_name.upper()} dataset")
    print(f"{'='*80}\n")

    # Get dataset config
    config = DATASET_CONFIGS[dataset_name]

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / dataset_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = config['class'](
        data_root=config['data_path'],
        split=config['train_split'],
        img_size=args.img_size,
        use_augmentation=True,
        normalize=True
    )

    val_dataset = config['class'](
        data_root=config['data_path'],
        split=config['val_split'],
        img_size=args.img_size,
        use_augmentation=False,
        normalize=True
    )

    num_classes = config['num_classes']

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    # Load backbone and create model
    print("Creating model...")
    backbone = load_backbone(args.backbone, args.pretrained_weights)
    model = DINOv3SegmentationModel(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout
    )
    model = model.to(device)

    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")

    # Calculate class weights
    if args.use_class_weights:
        print("Calculating class weights...")
        class_weights = train_dataset.get_class_weights().to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training with early stopping
    epochs = config.get('epochs', args.epochs)
    print(f"\nTraining for up to {epochs} epoch(s) with early stopping (patience=50)...")
    start_time = time.time()

    # Early stopping variables
    best_miou = 0.0
    patience = 50  # Stop if no improvement for 50 epochs
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        val_metrics, val_preds, val_targets = evaluate(model, val_loader, criterion, num_classes, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save predictions for visualization
        np.save(output_dir / 'val_predictions.npy', val_preds)
        np.save(output_dir / 'val_targets.npy', val_targets)

        # Check for improvement
        current_miou = val_metrics['miou']
        if current_miou > best_miou:
            best_miou = current_miou
            patience_counter = 0
            # Save best model
            # Handle DataParallel wrapper
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }
            torch.save(best_model_state, output_dir / 'model.pth')
            print(f"✓ New best mIoU: {best_miou:.4f} - Model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs (Best mIoU: {best_miou:.4f})")

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n早停触发！连续 {patience} 个epoch验证集mIoU未提升")
            print(f"最佳mIoU: {best_miou:.4f} (Epoch {epoch - patience + 1})")
            break

        # Save current metrics
        results = {
            'dataset': dataset_name,
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_miou': best_miou,
            'patience_counter': patience_counter,
            'training_time': time.time() - start_time,
            'early_stopped': patience_counter >= patience
        }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser('Quick 1-Epoch Training for All Datasets')

    # Dataset parameters
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        help='Datasets to train on (default: all)')
    parser.add_argument('--img_size', default=512, type=int,
                        help='Image size for training')

    # Model parameters
    parser.add_argument('--backbone', default='dinov3_vitl16', type=str,
                        help='Backbone architecture')
    parser.add_argument('--pretrained_weights',
                        default=None,
                        type=str, help='Path to pretrained weights (e.g., dinov3_vitl16_pretrain.pth)')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone weights')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate in segmentation head')

    # Training parameters
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for loss')

    # Output
    parser.add_argument('--output_dir',
                        default=str(PROJECT_ROOT / 'trained_models'),
                        type=str, help='Path to save outputs')

    # Misc
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Determine which datasets to train on
    if 'all' in args.datasets:
        datasets_to_train = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_train = [d for d in args.datasets if d in DATASET_CONFIGS]

    print(f"Training on datasets: {datasets_to_train}")

    # Train on each dataset
    all_results = {}
    for dataset_name in datasets_to_train:
        try:
            results = train_dataset(dataset_name, args)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\nError training on {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_path = Path(args.output_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    for dataset_name, results in all_results.items():
        val_metrics = results['val_metrics']
        print(f"{dataset_name:15s} - mIoU: {val_metrics['miou']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
