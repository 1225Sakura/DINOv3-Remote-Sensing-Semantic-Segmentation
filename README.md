# DINOv3 Remote Sensing Semantic Segmentation

A comprehensive semantic segmentation framework for remote sensing imagery using DINOv3 (Vision Transformer) as the backbone. This project supports training and inference on six major remote sensing datasets.

## Features

- **Multi-Dataset Support**: Train on 6 different remote sensing datasets
- **DINOv3 Backbone**: Leverages powerful vision transformer features
- **Easy to Use**: Simple training and inference scripts
- **Production Ready**: Clean, organized codebase suitable for research and production

## Supported Datasets

| Dataset | Classes | Task | Image Size |
|---------|---------|------|------------|
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 7 | Land-cover classification | Variable |
| [iSAID](https://captain-whu.github.io/iSAID/) | 16 | Aerial scene understanding | Variable |
| [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 5 | Urban semantic labeling | Variable |
| [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 | Urban semantic labeling | Variable |
| [LandCover.ai](https://landcover.ai/) | 5 | Land cover classification | 512x512 |
| [OpenEarthMap](https://open-earth-map.org/) | 8 | Global land cover mapping | Variable |

## Project Structure

```
remote_sensing_segmentation/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
│
├── datasets/                  # Dataset loaders
│   ├── __init__.py
│   ├── loveda_dataset.py
│   ├── isaid_dataset.py
│   ├── vaihingen_dataset.py
│   ├── potsdam_dataset.py
│   ├── landcoverai_dataset.py
│   ├── openearthmap_dataset.py
│   ├── LoveDA/               # Dataset files (not in git)
│   ├── iSAID/                # Dataset files (not in git)
│   ├── Vaihingen/            # Dataset files (not in git)
│   ├── Potsdam/              # Dataset files (not in git)
│   ├── LandCoverai/          # Dataset files (not in git)
│   └── OpenEarthMap/         # Dataset files (not in git)
│
├── models/                    # Model architecture
│   ├── __init__.py
│   └── segmentation_model.py
│
├── scripts/                   # Training and inference
│   ├── train.py              # Main training script
│   └── generate_predictions.py
│
├── tests/                     # Unit tests
│   └── test_dataset.py
│
├── docs/                      # Documentation
│   ├── DATASET_SETUP.md      # Dataset setup guide
│   ├── TRAINING_RESULTS.md   # Training results
│   └── FINAL_COMPREHENSIVE_REPORT.md
│
└── configs/                   # Configuration files
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- DINOv3 repository

### Setup

1. Clone the DINOv3 repository and this project:

```bash
# Clone DINOv3
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3

# Clone this repository into dinov3/
git clone <this-repo-url> remote_sensing_segmentation
cd remote_sensing_segmentation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download datasets and models:

**Due to GitHub file size limitations, datasets and trained models are hosted on Baidu Cloud.**

📦 **Baidu Cloud Download**: https://pan.baidu.com/s/5CXLX9bODEHBSVfKVRLsmdg

The cloud storage contains:
- `model/` - Trained model files (~7.2GB) for all 6 datasets
- `数据集/` - Complete datasets (optional, can also download from official sources)

For detailed download instructions and file structure, see:
- [Download Guide (English)](docs/LARGE_FILES.md)
- [下载说明 (中文)](docs/LARGE_FILES_CN.md)

Alternatively, you can download the original datasets yourself and set them up following the [Dataset Setup Guide](docs/DATASET_SETUP.md)

## Quick Start

### Training

Train on a single dataset:

```bash
cd /path/to/dinov3
python remote_sensing_segmentation/scripts/train.py \
    --datasets loveda \
    --batch_size 4 \
    --num_workers 4
```

Train on multiple datasets:

```bash
python remote_sensing_segmentation/scripts/train.py \
    --datasets loveda isaid vaihingen \
    --batch_size 4 \
    --num_workers 4
```

Train on all datasets:

```bash
python remote_sensing_segmentation/scripts/train.py \
    --datasets all \
    --batch_size 4 \
    --num_workers 4
```

### Inference

Generate predictions for a trained model:

```bash
python remote_sensing_segmentation/scripts/generate_predictions.py \
    --datasets loveda \
    --models_dir trained_models/quick_train \
    --output_dir predictions
```

## Configuration

### Training Parameters

- `--datasets`: Which datasets to train on (`all` or specific names)
- `--batch_size`: Batch size for training (default: 4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--img_size`: Image size for training (default: 512)
- `--backbone`: DINOv3 backbone variant (default: dinov3_vitl16)

### Dataset-Specific Settings

Different datasets can use different numbers of epochs. See `scripts/train.py` for configuration details.

## Model Architecture

The segmentation model uses:
- **Backbone**: DINOv3 Vision Transformer (frozen or fine-tuned)
- **Head**: Lightweight segmentation decoder
- **Loss**: Cross-entropy with optional class weights

## Results

Training results and performance metrics can be found in:
- [Training Results](docs/TRAINING_RESULTS.md)
- [Comprehensive Report](docs/FINAL_COMPREHENSIVE_REPORT.md)

## Testing

Run tests to verify dataset loading:

```bash
python tests/test_dataset.py
```

## Project Dependencies

This project depends on the DINOv3 repository for the backbone model. Make sure to:
1. Clone DINOv3 first
2. Place this project inside the DINOv3 directory
3. Follow the DINOv3 setup instructions for downloading pretrained weights

## License

This project is released under the same license as DINOv3. Please refer to the original DINOv3 repository for license details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- DINOv3 team at Meta AI for the excellent vision transformer backbone
- Dataset providers for making their data publicly available
- PyTorch and the open-source community
