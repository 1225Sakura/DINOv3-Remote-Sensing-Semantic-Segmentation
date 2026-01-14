#!/usr/bin/env python3
# Quick test to verify training setup

import sys
import os
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("Quick Training Test")
print("="*60)

# Test 1: Load dataset
print("\n[1] Loading dataset...")
from datasets.loveda_dataset import LoveDADataset

dataset = LoveDADataset(
    data_root=str(PROJECT_ROOT / "datasets" / "LoveDA"),  # Adjust path as needed
    split='Train',
    img_size=512,
    use_augmentation=False
)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Test 2: Load backbone
print("\n[2] Loading backbone...")
try:
    import dinov3.hub.backbones as backbones
    backbone = backbones.dinov3_vit7b16(pretrained=False)
    print(f"✓ Backbone loaded")
    print(f"  embed_dim: {backbone.embed_dim}")
    print(f"  patch_size: {backbone.patch_size}")
except ImportError:
    print("⚠ DINOv3 not found. Please install from https://github.com/facebookresearch/dinov3")
    sys.exit(1)

# Test 3: Load pretrained weights (optional)
print("\n[3] Loading pretrained weights...")
weights_path = str(PROJECT_ROOT / "checkpoints" / "dinov3_vit7b16_pretrain.pth")
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace('teacher.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"✓ Weights loaded")
    print(f"  Missing keys: {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
else:
    print(f"⚠ Pretrained weights not found at: {weights_path}")

# Test 4: Create segmentation model
print("\n[4] Creating segmentation model...")
from models.segmentation_model import DINOv3SegmentationModel

model = DINOv3SegmentationModel(
    backbone=backbone,
    num_classes=7,
    freeze_backbone=True,
    dropout=0.1
)
print(f"✓ Model created")

# Test 5: Test forward pass
print("\n[5] Testing forward pass...")
img, mask = dataset[0]
img = img.unsqueeze(0)  # Add batch dimension

if torch.cuda.is_available():
    print("  Using CUDA")
    model = model.cuda()
    img = img.cuda()

with torch.no_grad():
    output = model(img)

print(f"✓ Forward pass successful")
print(f"  Input shape: {img.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Expected shape: (1, 7, 512, 512)")

# Test 6: Test loss calculation
print("\n[6] Testing loss calculation...")
target = mask.unsqueeze(0)
if torch.cuda.is_available():
    target = target.cuda()

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, target)
print(f"✓ Loss calculated: {loss.item():.4f}")

print("\n" + "="*60)
print("✓ All tests passed! Ready to train.")
print("="*60)
