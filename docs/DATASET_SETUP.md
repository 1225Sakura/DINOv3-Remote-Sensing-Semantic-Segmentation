# Dataset Setup Guide

This document provides detailed instructions for setting up the six remote sensing semantic segmentation datasets used in this project.

## Supported Datasets

1. **LoveDA** - Land-cover dataset for domain adaptive semantic segmentation
2. **iSAID** - Instance segmentation dataset in aerial images
3. **Vaihingen** - ISPRS 2D semantic labeling contest
4. **Potsdam** - ISPRS 2D semantic labeling contest
5. **LandCover.ai** - High resolution land cover classification
6. **OpenEarthMap** - Global land cover mapping dataset

## Dataset Structure

All datasets should be placed in the `datasets/` directory with the following structure:

```
datasets/
├── __init__.py
├── loveda_dataset.py
├── isaid_dataset.py
├── vaihingen_dataset.py
├── potsdam_dataset.py
├── landcoverai_dataset.py
├── openearthmap_dataset.py
├── LoveDA/
├── iSAID/
├── Vaihingen/
├── Potsdam/
├── LandCoverai/
└── OpenEarthMap/
```

## 1. LoveDA Dataset

**Download:** https://github.com/Junjue-Wang/LoveDA

**Expected Structure:**
```
LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    │   ├── images_png/
    │   └── masks_png/
    └── Rural/
        ├── images_png/
        └── masks_png/
```

**Classes:** 7 classes (background, building, road, water, barren, forest, agricultural)

## 2. iSAID Dataset

**Download:** https://captain-whu.github.io/iSAID/

**Expected Structure:**
```
iSAID/
├── train/
│   ├── images/
│   └── Semantic_masks/
│       └── images/
└── val/
    ├── images/
    └── Semantic_masks/
        └── images/
```

**Classes:** 16 classes (background, ship, storage_tank, baseball_diamond, tennis_court, basketball_court, ground_track_field, bridge, large_vehicle, small_vehicle, helicopter, swimming_pool, roundabout, soccer_ball_field, plane, harbor)

**Notes:**
- Masks are RGB encoded images
- Original images should be placed in `images/` folder
- Semantic masks should be placed in `Semantic_masks/images/` folder

## 3. Vaihingen Dataset

**Download:** https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx

**Expected Structure:**
```
Vaihingen/
├── train/
│   ├── train_image_rgb/
│   └── train_label_rgb/
└── test/
    ├── test_image_rgb/
    └── test_label_rgb/
```

**Classes:** 5 classes (impervious surfaces, building, low vegetation, tree, car)

**Notes:**
- Images can be either .tif or .png format
- Labels are RGB encoded

## 4. Potsdam Dataset

**Download:** https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

**Expected Structure:**
```
Potsdam/
├── train/
│   ├── train_image_rgb/
│   └── train_label_rgb/
└── test/
    ├── test_image_rgb/
    └── test_label_rgb/
```

**Classes:** 6 classes (impervious surfaces, building, low vegetation, tree, car, clutter/background)

## 5. LandCover.ai Dataset

**Download:** https://landcover.ai/

**Expected Structure:**
```
LandCoverai/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

**Classes:** 5 classes (background, buildings, woodlands, water, roads)

## 6. OpenEarthMap Dataset

**Download:** https://open-earth-map.org/

**Expected Structure:**
```
OpenEarthMap/
├── Train/
│   ├── images/
│   └── labels/
└── Val/
    ├── images/
    └── labels/
```

**Classes:** 8 classes (bareland, grass, pavement, road, tree, water, cropland, building)

**Notes:**
- Images are in TIFF format
- Uses non-contiguous label values (0,2,3,4,5,6,7,8) which are automatically mapped to contiguous (0-7)
- Requires `tifffile` library for reading TIFF images

## Setup Instructions

1. Create the datasets directory:
   ```bash
   mkdir -p datasets
   ```

2. Download each dataset from the links above

3. Extract and organize them according to the structure shown

4. Verify dataset loading:
   ```bash
   python tests/test_dataset.py
   ```

## Notes

- All datasets use different image sizes and formats
- Images are automatically resized to 512x512 during training
- Augmentation is applied during training for better generalization
- RGB-encoded masks are automatically converted to class IDs
