# Copyright (c) Remote Sensing Segmentation Training
# LandCover.ai Dataset Loader for Semantic Segmentation

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LandCoveraiDataset(Dataset):
    """
    LandCover.ai Dataset for remote sensing semantic segmentation.

    Dataset structure:
        data_root/
            train/
                image/
                    *.jpg
                label/
                    *_m.png
            val/
                image/
                    *.jpg
                label/
                    *_m.png

    Classes: Background, Building, Woodland, Water, Road
    """

    CLASSES = [
        'background',
        'building',
        'woodland',
        'water',
        'road'
    ]

    NUM_CLASSES = 5

    # LandCover.ai uses grayscale labels with values 0-4
    # Visualization palette
    PALETTE = [
        [0, 0, 0],        # background - black
        [255, 0, 0],      # building - red
        [0, 255, 0],      # woodland - green
        [0, 0, 255],      # water - blue
        [255, 255, 0],    # road - yellow
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
            data_root: Root directory of LandCover.ai dataset
            split: 'train' or 'val'
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
        img_dir = os.path.join(data_root, split, 'image')
        label_dir = os.path.join(data_root, split, 'label')

        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.mask_paths = []

        for img_path in self.image_paths:
            img_name = os.path.basename(img_path).replace('.jpg', '_m.png')
            mask_path = os.path.join(label_dir, img_name)
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)

        # Only keep image paths that have corresponding masks
        valid_indices = [i for i, mask_path in enumerate(self.mask_paths) if mask_path]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]

        print(f"LandCover.ai {split}: Found {len(self.image_paths)} images")

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))

        # Load mask - LandCover.ai uses grayscale masks with pixel values representing class IDs
        mask = np.array(Image.open(self.mask_paths[idx]))

        # If mask is RGB, convert to grayscale by taking first channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Ensure mask values are in valid range [0, NUM_CLASSES-1]
        mask = mask.astype(np.int64)
        mask[mask >= self.NUM_CLASSES] = 0  # Map invalid values to background

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
        for class_id in range(LandCoveraiDataset.NUM_CLASSES):
            rgb_mask[mask == class_id] = LandCoveraiDataset.PALETTE[class_id]
        return rgb_mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        print("Calculating class weights for LandCover.ai...")
        class_counts = np.zeros(self.NUM_CLASSES)

        # Sample subset for speed (every 10th image)
        sample_paths = self.mask_paths[::10] if len(self.mask_paths) > 100 else self.mask_paths

        for mask_path in sample_paths:
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

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
