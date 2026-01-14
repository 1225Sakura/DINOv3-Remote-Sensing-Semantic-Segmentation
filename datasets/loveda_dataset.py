# Copyright (c) Remote Sensing Segmentation Training
# LoveDA Dataset Loader for Semantic Segmentation

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LoveDADataset(Dataset):
    """
    LoveDA Dataset for remote sensing semantic segmentation.

    Dataset structure:
        data_root/
            Train/
                Urban/
                    images_png/
                    masks_png/
                Rural/
                    images_png/
                    masks_png/
            Val/
                Urban/
                    images_png/
                    masks_png/
                Rural/
                    images_png/
                    masks_png/

    Classes: Background, Building, Road, Water, Barren, Forest, Agricultural
    """

    CLASSES = [
        'background',
        'building',
        'road',
        'water',
        'barren',
        'forest',
        'agricultural'
    ]

    NUM_CLASSES = 7

    # Color palette for visualization (RGB)
    PALETTE = [
        [255, 255, 255],  # background - white
        [255, 0, 0],      # building - red
        [255, 255, 0],    # road - yellow
        [0, 0, 255],      # water - blue
        [159, 129, 183],  # barren - purple
        [0, 255, 0],      # forest - green
        [255, 195, 128],  # agricultural - orange
    ]

    def __init__(
        self,
        data_root,
        split='Train',
        img_size=512,
        use_augmentation=True,
        normalize=True
    ):
        """
        Args:
            data_root: Root directory of LoveDA dataset
            split: 'Train' or 'Val'
            img_size: Target image size (will be resized to this)
            use_augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and (split == 'Train')
        self.normalize = normalize

        # Collect all image paths
        self.image_paths = []
        self.mask_paths = []

        for region in ['Urban', 'Rural']:
            img_dir = os.path.join(data_root, split, region, 'images_png')
            mask_dir = os.path.join(data_root, split, region, 'masks_png')

            if os.path.exists(img_dir):
                images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    mask_path = os.path.join(mask_dir, img_name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        print(f"LoveDA {split}: Found {len(self.image_paths)} images")

        # Setup augmentation
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup augmentation and normalization transforms"""
        if self.use_augmentation:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=self.img_size, width=self.img_size, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                ], p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                ], p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
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
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))

        # LoveDA labels are 1-7, we need to map to 0-6
        # Label 0 in data means "ignore/unlabeled", map to 255 (ignore_index)
        mask = mask.astype(np.int64)
        mask[mask == 0] = 255  # ignore unlabeled pixels
        mask[mask != 255] = mask[mask != 255] - 1  # shift 1-7 to 0-6

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert mask to long tensor if not already a tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask

    @staticmethod
    def decode_target(mask):
        """Decode segmentation mask to RGB for visualization"""
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id in range(LoveDADataset.NUM_CLASSES):
            rgb_mask[mask == class_id] = LoveDADataset.PALETTE[class_id]
        return rgb_mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        print("Calculating class weights...")
        class_counts = np.zeros(self.NUM_CLASSES)

        for mask_path in self.mask_paths:
            mask = np.array(Image.open(mask_path))
            # LoveDA labels are 1-7, map to 0-6 for counting
            for class_id in range(self.NUM_CLASSES):
                # Original label is class_id + 1 (1-7 maps to 0-6)
                class_counts[class_id] += np.sum(mask == (class_id + 1))

        # Inverse frequency weighting
        total_pixels = np.sum(class_counts)
        class_weights = total_pixels / (self.NUM_CLASSES * class_counts + 1e-8)

        # Normalize
        class_weights = class_weights / class_weights.sum() * self.NUM_CLASSES

        print("Class counts:", class_counts)
        print("Class weights:", class_weights)

        return torch.FloatTensor(class_weights)


if __name__ == "__main__":
    # Test the dataset
    dataset = LoveDADataset(
        data_root="./datasets/LoveDA",  # Adjust path as needed
        split="Train",
        img_size=512,
        use_augmentation=True
    )

    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
