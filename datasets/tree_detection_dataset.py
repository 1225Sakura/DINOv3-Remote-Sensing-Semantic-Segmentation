#!/usr/bin/env python3
"""
Tree Detection Dataset Loader
Loads tree detection data in PASCAL VOC XML format
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TreeDetectionDataset(Dataset):
    """
    Dataset for tree detection with PASCAL VOC XML format labels

    Args:
        data_root: Root directory containing 'images' and 'labels' folders
        split: 'train' or 'val'
        img_size: Size to resize images to (default: 512)
        use_augmentation: Whether to apply data augmentation (default: True for train)
        normalize: Whether to normalize images (default: True)
        train_ratio: Ratio of training data (default: 0.8)
        random_seed: Random seed for split reproducibility (default: 42)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 512,
        use_augmentation: bool = True,
        normalize: bool = True,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation if split == 'train' else False
        self.normalize = normalize

        # Paths
        self.images_dir = self.data_root / 'images'
        self.labels_dir = self.data_root / 'labels'

        # Read map.txt to get all samples
        map_file = self.data_root / 'map.txt'
        if not map_file.exists():
            raise FileNotFoundError(f"map.txt not found at {map_file}")

        # Parse map.txt
        all_samples = []
        with open(map_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_path = parts[0].replace('images\\', '').replace('images/', '')
                    label_path = parts[1].replace('labels\\', '').replace('labels/', '')
                    all_samples.append((img_path, label_path))

        # Split data into train/val
        np.random.seed(random_seed)
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)

        # Shuffle and split
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        if split == 'train':
            self.samples = [all_samples[i] for i in train_indices]
        elif split == 'val':
            self.samples = [all_samples[i] for i in val_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Setup transformations
        self._setup_transforms()

        # Class info
        self.num_classes = 1  # Only 'tree' class
        self.class_names = ['tree']

    def _setup_transforms(self):
        """Setup image transformations"""

        # Base transforms (always applied)
        base_transforms = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=0,
                value=0
            ),
        ]

        # Augmentation transforms (only for training)
        aug_transforms = []
        if self.use_augmentation:
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussianBlur(p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
            ]

        # Normalization
        norm_transforms = []
        if self.normalize:
            # ImageNet normalization (standard for vision transformers)
            norm_transforms = [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                )
            ]

        # Combine all transforms
        all_transforms = base_transforms + aug_transforms + norm_transforms + [ToTensorV2()]

        self.transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )

    def _parse_xml(self, xml_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Parse PASCAL VOC XML file

        Returns:
            bboxes: List of [xmin, ymin, xmax, ymax] bounding boxes
            labels: List of class labels (all 0 for 'tree')
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        bboxes = []
        labels = []

        # Parse each object
        for obj in root.findall('object'):
            # Get bbox
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Clip to image boundaries
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))

            # Only add valid boxes
            if xmax > xmin and ymax > ymin:
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(0)  # All trees are class 0

        return bboxes, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample

        Returns:
            image: Tensor of shape (C, H, W)
            target: Dict containing:
                - boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2] format
                - labels: Tensor of shape (N,) with class labels
                - image_id: Original image filename
                - orig_size: Original image size (H, W)
        """
        img_name, label_name = self.samples[idx]

        # Load image
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        orig_h, orig_w = image.shape[:2]

        # Load annotations
        label_path = self.labels_dir / label_name
        bboxes, labels = self._parse_xml(label_path)

        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            # No boxes - still transform image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = []
            labels = []

        # Convert to tensors
        if len(bboxes) > 0:
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_name,
            'orig_size': torch.as_tensor([orig_h, orig_w])
        }

        return image, target

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        total_boxes = 0
        boxes_per_image = []

        for idx in range(len(self)):
            _, label_name = self.samples[idx]
            label_path = self.labels_dir / label_name
            bboxes, _ = self._parse_xml(label_path)
            n_boxes = len(bboxes)
            total_boxes += n_boxes
            boxes_per_image.append(n_boxes)

        return {
            'num_images': len(self),
            'total_trees': total_boxes,
            'avg_trees_per_image': np.mean(boxes_per_image),
            'min_trees_per_image': np.min(boxes_per_image),
            'max_trees_per_image': np.max(boxes_per_image),
        }


def collate_fn(batch):
    """
    Custom collate function for batching detection data

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: Batched images tensor (B, C, H, W)
        targets: List of target dicts (one per image)
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


# Test code
if __name__ == '__main__':
    # Test dataset loading
    data_root = './datasets/tree_detection'  # Adjust path as needed

    print("Creating train dataset...")
    train_dataset = TreeDetectionDataset(
        data_root=data_root,
        split='train',
        img_size=512,
        use_augmentation=True
    )

    print("Creating val dataset...")
    val_dataset = TreeDetectionDataset(
        data_root=data_root,
        split='val',
        img_size=512,
        use_augmentation=False
    )

    print("\nTrain dataset statistics:")
    train_stats = train_dataset.get_statistics()
    for key, value in train_stats.items():
        print(f"  {key}: {value}")

    print("\nVal dataset statistics:")
    val_stats = val_dataset.get_statistics()
    for key, value in val_stats.items():
        print(f"  {key}: {value}")

    print("\nTesting data loading...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    images, targets = next(iter(train_loader))
    print(f"\nBatch loaded successfully!")
    print(f"Images shape: {images.shape}")
    print(f"Number of targets: {len(targets)}")
    print(f"First target boxes shape: {targets[0]['boxes'].shape}")
    print(f"First target labels shape: {targets[0]['labels'].shape}")
