# Copyright (c) Remote Sensing Segmentation Training
# Potsdam Dataset Loader for Semantic Segmentation

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PotsdamDataset(Dataset):
    """
    ISPRS Potsdam Dataset for remote sensing semantic segmentation.

    Dataset structure:
        data_root/
            train_image/
                *.jpg
            train_label/
                *_label.png
            test_image/
                *.jpg
            test_label/  (if available)
                *_label.png

    Classes: Impervious surfaces, Building, Low vegetation, Tree, Car, Clutter
    """

    CLASSES = [
        'impervious_surfaces',  # White - roads, parking
        'building',             # Blue
        'low_vegetation',       # Cyan - grass, small plants
        'tree',                 # Green - tall vegetation
        'car',                  # Yellow
        'clutter'               # Red - background/clutter
    ]

    NUM_CLASSES = 6

    # Potsdam uses RGB color coding for labels
    LABEL_COLORS = {
        (255, 255, 255): 0,  # Impervious surfaces - White
        (0, 0, 255): 1,      # Building - Blue
        (0, 255, 255): 2,    # Low vegetation - Cyan
        (0, 255, 0): 3,      # Tree - Green
        (255, 255, 0): 4,    # Car - Yellow
        (255, 0, 0): 5,      # Clutter/boundary - Red
    }

    # Visualization palette
    PALETTE = [
        [255, 255, 255],  # impervious_surfaces - white
        [0, 0, 255],      # building - blue
        [0, 255, 255],    # low_vegetation - cyan
        [0, 255, 0],      # tree - green
        [255, 255, 0],    # car - yellow
        [255, 0, 0],      # clutter - red
    ]

    def __init__(
        self,
        data_root,
        split='train',
        img_size=512,
        use_augmentation=True,
        normalize=True
    ):
        """
        Args:
            data_root: Root directory of Potsdam dataset
            split: 'train' or 'test'
            img_size: Target image size (will be resized to this)
            use_augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.normalize = normalize

        # Collect all image paths
        img_dir = os.path.join(data_root, f'{split}_image')
        label_dir = os.path.join(data_root, f'{split}_label')

        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.mask_paths = []

        for img_path in self.image_paths:
            img_name = os.path.basename(img_path).replace('_RGB.jpg', '_label.png')
            mask_path = os.path.join(label_dir, img_name)
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
            else:
                # For test set, labels might not exist
                self.mask_paths.append(None)

        print(f"Potsdam {split}: Found {len(self.image_paths)} images")

        # Setup augmentation
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup augmentation and normalization transforms"""
        if self.use_augmentation:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=self.img_size, width=self.img_size, scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])

    def _rgb_to_class_id(self, rgb_mask):
        """Convert RGB mask to class ID mask"""
        h, w = rgb_mask.shape[:2]
        class_mask = np.full((h, w), 255, dtype=np.uint8)  # Default to ignore

        for color, class_id in self.LABEL_COLORS.items():
            matches = np.all(rgb_mask == color, axis=-1)
            class_mask[matches] = class_id

        return class_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))

        # Load mask if available
        if self.mask_paths[idx] is not None:
            rgb_mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
            mask = self._rgb_to_class_id(rgb_mask)
        else:
            # Create dummy mask for test set
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert mask to long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask

    @staticmethod
    def decode_target(mask):
        """Decode segmentation mask to RGB for visualization"""
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id in range(PotsdamDataset.NUM_CLASSES):
            rgb_mask[mask == class_id] = PotsdamDataset.PALETTE[class_id]
        return rgb_mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        print("Calculating class weights for Potsdam...")
        class_counts = np.zeros(self.NUM_CLASSES)

        # Sample subset for speed
        sample_paths = self.mask_paths[::10]  # Sample every 10th image

        for mask_path in sample_paths:
            if mask_path is None:
                continue
            rgb_mask = np.array(Image.open(mask_path).convert('RGB'))
            mask = self._rgb_to_class_id(rgb_mask)

            for class_id in range(self.NUM_CLASSES):
                class_counts[class_id] += np.sum(mask == class_id)

        # Inverse frequency weighting
        total_pixels = np.sum(class_counts)
        class_weights = total_pixels / (self.NUM_CLASSES * class_counts + 1e-8)

        # Normalize
        class_weights = class_weights / class_weights.sum() * self.NUM_CLASSES

        print("Class counts (sampled):", class_counts)
        print("Class weights:", class_weights)

        return torch.FloatTensor(class_weights)
