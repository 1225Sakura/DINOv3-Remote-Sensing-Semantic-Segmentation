#!/usr/bin/env python3
# Copyright (c) Remote Sensing Segmentation Training
# Generate prediction results with original filenames

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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

# Dataset configurations
DATASET_CONFIGS = {
    'loveda': {
        'class': LoveDADataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'LoveDA',
        'val_split': 'Val',
        'num_classes': 7,
        'palette': LoveDADataset.PALETTE
    },
    'isaid': {
        'class': iSAIDDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'iSAID',
        'val_split': 'val',
        'num_classes': 16,
        'palette': iSAIDDataset.PALETTE
    },
    'vaihingen': {
        'class': VaihingenDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'Vaihingen',
        'val_split': 'test',
        'num_classes': 5,
        'palette': VaihingenDataset.PALETTE
    },
    'potsdam': {
        'class': PotsdamDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'Potsdam',
        'val_split': 'test',
        'num_classes': 6,
        'palette': PotsdamDataset.PALETTE
    },
    'landcoverai': {
        'class': LandCoveraiDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'LandCoverai',
        'val_split': 'val',
        'num_classes': 5,
        'palette': LandCoveraiDataset.PALETTE
    },
    'openearthmap': {
        'class': OpenEarthMapDataset,
        'data_path': PROJECT_ROOT / 'datasets' / 'OpenEarthMap',
        'val_split': 'Val',
        'num_classes': 8,
        'palette': OpenEarthMapDataset.PALETTE
    }
}


def load_backbone(backbone_name):
    """Load DINOv3 backbone"""
    import dinov3.hub.backbones as backbones
    backbone_fn = getattr(backbones, backbone_name)
    backbone = backbone_fn(pretrained=False)
    return backbone


def decode_segmentation_mask(mask, palette):
    """Convert segmentation mask to RGB image"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(palette):
        rgb_mask[mask == class_id] = color

    return rgb_mask


def get_original_filename(dataset, idx):
    """Extract original filename from dataset path"""
    if hasattr(dataset, 'image_paths'):
        img_path = dataset.image_paths[idx]
    elif hasattr(dataset, 'img_paths'):
        img_path = dataset.img_paths[idx]
    else:
        return f"pred_{idx:04d}.png"

    # Get basename without extension
    basename = Path(img_path).stem

    # Remove common suffixes
    suffixes_to_remove = ['_instance_color_RGB', '_instance_id_RGB']
    for suffix in suffixes_to_remove:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]

    return basename + '.png'


def generate_predictions(dataset_name, args):
    """Generate prediction images for entire val/test set"""
    print(f"\n{'='*80}")
    print(f"Generating predictions for {dataset_name.upper()}")
    print(f"{'='*80}\n")

    config = DATASET_CONFIGS[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the most recent model directory
    model_base_dir = Path(args.models_dir) / dataset_name
    if not model_base_dir.exists():
        print(f"❌ No model directory found for {dataset_name}")
        return

    # Get the most recent training run
    model_dirs = sorted([d for d in model_base_dir.iterdir() if d.is_dir()],
                       key=lambda x: x.name, reverse=True)

    if not model_dirs:
        print(f"❌ No model found for {dataset_name}")
        return

    model_dir = model_dirs[0]
    model_path = model_dir / 'model.pth'

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"Loading model from: {model_path}")

    # Create output directory
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    val_dataset = config['class'](
        data_root=config['data_path'],
        split=config['val_split'],
        img_size=args.img_size,
        use_augmentation=False,
        normalize=True
    )

    print(f"Dataset size: {len(val_dataset)} images")

    # Load model
    backbone = load_backbone(args.backbone)
    model = DINOv3SegmentationModel(
        backbone=backbone,
        num_classes=config['num_classes'],
        freeze_backbone=True,
        dropout=0.1
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Generating predictions for all {len(val_dataset)} images...")

    # Generate predictions for all images
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc=f"Processing {dataset_name}"):
            image, _ = val_dataset[idx]

            # Add batch dimension and move to device
            image_batch = image.unsqueeze(0).to(device)

            # Predict
            output = model(image_batch)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Decode to RGB
            pred_rgb = decode_segmentation_mask(prediction, config['palette'])

            # Save prediction image with original filename
            original_filename = get_original_filename(val_dataset, idx)
            pred_img = Image.fromarray(pred_rgb)
            save_path = output_dir / original_filename
            pred_img.save(save_path)

    print(f"\n✅ Saved {len(val_dataset)} prediction images to: {output_dir}")


def main():
    parser = argparse.ArgumentParser('Generate Prediction Images with Original Filenames')

    parser.add_argument('--datasets', nargs='+', default=['all'],
                       help='Datasets to process (default: all)')
    parser.add_argument('--models_dir',
                       default=str(PROJECT_ROOT / 'trained_models'),
                       type=str, help='Directory containing trained models')
    parser.add_argument('--output_dir',
                       default=str(PROJECT_ROOT / 'predictions'),
                       type=str, help='Output directory for prediction images')
    parser.add_argument('--img_size', default=512, type=int,
                       help='Image size used during training')
    parser.add_argument('--backbone', default='dinov3_vitl16', type=str,
                       help='Backbone architecture')

    args = parser.parse_args()

    # Determine which datasets to process
    if 'all' in args.datasets:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_process = [d for d in args.datasets if d in DATASET_CONFIGS]

    print(f"Generating predictions for datasets: {datasets_to_process}")
    print(f"Output directory: {args.output_dir}")

    # Print dataset sizes
    print(f"\nDataset sizes:")
    for dataset_name in datasets_to_process:
        config = DATASET_CONFIGS[dataset_name]
        try:
            ds = config['class'](
                data_root=config['data_path'],
                split=config['val_split'],
                img_size=512,
                use_augmentation=False,
                normalize=True
            )
            print(f"  {dataset_name:15s}: {len(ds):4d} images")
        except:
            print(f"  {dataset_name:15s}: Error loading dataset")

    # Process each dataset
    for dataset_name in datasets_to_process:
        try:
            generate_predictions(dataset_name, args)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("✅ All predictions completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
