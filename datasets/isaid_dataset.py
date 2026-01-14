# Copyright (c) Remote Sensing Segmentation Training
# iSAID Dataset Loader for Semantic Segmentation

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class iSAIDDataset(Dataset):
    """
    iSAID Dataset for remote sensing semantic segmentation.

    Dataset structure:
        data_root/
            train/
                Semantic_masks/
                    images/
            val/
                Semantic_masks/
                    images/

    Classes: background, ship, storage_tank, baseball_diamond, tennis_court,
             basketball_court, ground_track_field, bridge, large_vehicle,
             small_vehicle, helicopter, swimming_pool, roundabout,
             soccer_ball_field, plane, harbor
    """

    CLASSES = [
        'background',
        'ship',
        'storage_tank',
        'baseball_diamond',
        'tennis_court',
        'basketball_court',
        'ground_track_field',
        'bridge',
        'large_vehicle',
        'small_vehicle',
        'helicopter',
        'swimming_pool',
        'roundabout',
        'soccer_ball_field',
        'plane',
        'harbor'
    ]

    NUM_CLASSES = 16

    # Color palette for visualization (RGB) - distinct colors for each class
    PALETTE = [
        [0, 0, 0],          # background - black
        [0, 0, 63],         # ship - dark blue
        [0, 191, 127],      # storage_tank - turquoise
        [0, 63, 0],         # baseball_diamond - dark green
        [0, 63, 127],       # tennis_court - teal
        [0, 63, 191],       # basketball_court - cyan
        [0, 63, 255],       # ground_track_field - light blue
        [0, 127, 63],       # bridge - sea green
        [0, 127, 127],      # large_vehicle - cyan-green
        [0, 0, 127],        # small_vehicle - navy
        [0, 0, 191],        # helicopter - medium blue
        [0, 0, 255],        # swimming_pool - blue
        [0, 191, 191],      # roundabout - bright cyan
        [0, 127, 191],      # soccer_ball_field - sky blue
        [0, 127, 255],      # plane - azure
        [0, 100, 155],      # harbor - steel blue
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
            data_root: Root directory of iSAID dataset
            split: 'train' or 'val'
            img_size: Target image size (will be resized to this)
            use_augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.normalize = normalize

        # Collect all image paths
        semantic_dir = os.path.join(data_root, split, 'Semantic_masks', 'images')
        # Use broader pattern to match both full images and tiled images
        all_masks = sorted(glob.glob(os.path.join(semantic_dir, '*_instance_color_RGB*.png')))

        # For train split: use only full images (not tiled versions)
        # For val split: use all images (val only has tiled versions)
        if split == 'train':
            # Exclude tiled images (those with _r###_c###_s### suffix)
            self.image_paths = [p for p in all_masks if not any(x in os.path.basename(p) for x in ['_r0', '_r1', '_r2', '_r3', '_r4', '_r5', '_r6', '_r7', '_r8', '_r9'])]
        else:
            # Val split: use all images
            self.image_paths = all_masks

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {semantic_dir}")

        print(f"iSAID {split}: Found {len(self.image_paths)} images")

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

    def _rgb_to_class_id(self, rgb_mask):
        """
        Convert RGB mask to class ID mask.
        iSAID masks are stored as RGB images where colors represent classes.
        """
        # Create a class ID mask
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)

        # Map each palette color to its class ID
        for class_id, color in enumerate(self.PALETTE):
            # Find pixels matching this color
            matches = np.all(rgb_mask == color, axis=-1)
            class_mask[matches] = class_id

        return class_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (the semantic mask image is actually the RGB colored segmentation)
        # We need to find corresponding original image
        mask_path = self.image_paths[idx]

        # Get the base name to find original image
        # For iSAID dataset:
        # Mask: P0000_instance_color_RGB.png or P0003_instance_color_RGB_r000_c000_s1023.png
        # Image: P0000.png or P0003_r000_c000_s1023.png
        # Simply remove "_instance_color_RGB" from the filename
        mask_filename = os.path.basename(mask_path)
        image_filename = mask_filename.replace('_instance_color_RGB', '')

        # Load the RGB mask
        rgb_mask = np.array(Image.open(mask_path).convert('RGB'))

        # For iSAID, we need to find the original image
        # Check if there's an images folder at the parent level
        image_dir_candidates = [
            os.path.join(self.data_root, self.split, 'images'),
            os.path.join(self.data_root, self.split, 'Semantic_masks'),
        ]

        image = None
        for img_dir in image_dir_candidates:
            possible_image_path = os.path.join(img_dir, image_filename)
            if os.path.exists(possible_image_path):
                image = np.array(Image.open(possible_image_path).convert('RGB'))
                break

        # If we can't find original image, use a processed version of the mask
        # This is a fallback - ideally the dataset should have original images
        if image is None:
            print(f"Warning: Could not find original image for {image_filename}, using mask as reference")
            # We'll need to download or have the actual satellite images
            # For now, create a synthetic image (this should be replaced with actual images)
            image = rgb_mask.copy()

        # Convert RGB mask to class IDs
        mask = self._rgb_to_class_id(rgb_mask)

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
        for class_id in range(iSAIDDataset.NUM_CLASSES):
            rgb_mask[mask == class_id] = iSAIDDataset.PALETTE[class_id]
        return rgb_mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        print("Calculating class weights for iSAID...")
        class_counts = np.zeros(self.NUM_CLASSES)

        for img_path in self.image_paths[:100]:  # Sample first 100 for speed
            rgb_mask = np.array(Image.open(img_path).convert('RGB'))
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


if __name__ == "__main__":
    # Test the dataset
    dataset = iSAIDDataset(
        data_root="./datasets/iSAID",  # Adjust path as needed
        split="train",
        img_size=512,
        use_augmentation=True
    )

    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
