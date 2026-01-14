# Copyright (c) Remote Sensing Segmentation Training
# OpenEarthMap Dataset Loader for Semantic Segmentation

import os
import glob
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OpenEarthMapDataset(Dataset):
    """
    OpenEarthMap Dataset for remote sensing semantic segmentation.

    Dataset structure:
        data_root/
            val_image/
                *.tif
            val_label/
                *.tif

    Classes: Bareland, Rangeland, Developed Space, Road, Tree, Water, Agricultural Land, Building
    Label values: 0, 2, 3, 4, 5, 6, 7, 8 (8 classes)
    """

    CLASSES = [
        'bareland',      # 0
        'rangeland',     # 2
        'developed',     # 3
        'road',          # 4
        'tree',          # 5
        'water',         # 6
        'agricultural',  # 7
        'building'       # 8
    ]

    NUM_CLASSES = 8

    # Map original label values to contiguous 0-7
    LABEL_MAPPING = {
        0: 0,   # bareland
        2: 1,   # rangeland
        3: 2,   # developed
        4: 3,   # road
        5: 4,   # tree
        6: 5,   # water
        7: 6,   # agricultural
        8: 7,   # building
    }

    # Color palette for visualization (RGB)
    PALETTE = [
        [128, 128, 128],  # bareland - gray
        [152, 251, 152],  # rangeland - pale green
        [192, 192, 192],  # developed - light gray
        [128, 64, 128],   # road - purple
        [0, 128, 0],      # tree - green
        [0, 0, 255],      # water - blue
        [255, 255, 0],    # agricultural - yellow
        [255, 0, 0],      # building - red
    ]

    def __init__(
        self,
        data_root,
        split='Val',
        img_size=512,
        use_augmentation=True,
        normalize=True
    ):
        """
        Args:
            data_root: Root directory of OpenEarthMap dataset
            split: 'Val' (only validation set available)
            img_size: Target image size (will be resized to this)
            use_augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_root = data_root
        self.split = split.lower()
        self.img_size = img_size
        self.use_augmentation = use_augmentation and (self.split == 'train')
        self.normalize = normalize

        # Collect all image paths
        self.image_paths = []
        self.mask_paths = []

        # For validation set
        img_dir = os.path.join(data_root, 'val_image')
        mask_dir = os.path.join(data_root, 'val_label')

        if os.path.exists(img_dir):
            images = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
            for img_path in images:
                img_name = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, img_name)

                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        print(f"OpenEarthMap {split}: Found {len(self.image_paths)} images")

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
        # Load image and mask using tifffile (handles special TIFF formats better than PIL)
        try:
            image = tifffile.imread(self.image_paths[idx])
            mask = tifffile.imread(self.mask_paths[idx])
        except Exception as e:
            # Fallback to PIL if tifffile fails
            try:
                image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
                mask = np.array(Image.open(self.mask_paths[idx]))
            except Exception as pil_error:
                raise RuntimeError(f"Failed to load image {self.image_paths[idx]}: {str(e)} / {str(pil_error)}")

        # Handle different data types and ensure correct format
        # Convert int16 to uint8 if necessary
        if image.dtype == np.int16:
            # Normalize int16 to [0, 255] range
            image = np.clip(image, 0, None)  # Remove negative values
            image = (image / image.max() * 255).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)

        # Ensure image has 3 channels
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] != 3:
            image = image[:, :, :3]

        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Map original label values (0, 2, 3, 4, 5, 6, 7, 8) to contiguous (0-7)
        mask_mapped = np.zeros_like(mask, dtype=np.int64) + 255  # Initialize with ignore_index
        for orig_val, new_val in self.LABEL_MAPPING.items():
            mask_mapped[mask == orig_val] = new_val

        # Apply transforms
        transformed = self.transform(image=image, mask=mask_mapped)
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
        for class_id in range(OpenEarthMapDataset.NUM_CLASSES):
            rgb_mask[mask == class_id] = OpenEarthMapDataset.PALETTE[class_id]
        return rgb_mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        print("Calculating class weights...")
        class_counts = np.zeros(self.NUM_CLASSES)

        for mask_path in self.mask_paths:
            mask = np.array(Image.open(mask_path))
            # Count pixels for each original label value
            for orig_val, new_val in self.LABEL_MAPPING.items():
                class_counts[new_val] += np.sum(mask == orig_val)

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
    dataset = OpenEarthMapDataset(
        data_root="./datasets/OpenEarthMap",  # Adjust path as needed
        split="Val",
        img_size=512,
        use_augmentation=False
    )

    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
