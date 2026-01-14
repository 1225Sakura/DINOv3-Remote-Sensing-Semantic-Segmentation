# DINOv3 遥感图像语义分割

一个基于DINOv3 (Vision Transformer) 作为骨干网络的综合遥感图像语义分割框架。本项目支持在六个主要遥感数据集上进行训练和推理。

## 特性

- **多数据集支持**: 支持6个不同的遥感数据集训练
- **DINOv3骨干网络**: 利用强大的视觉Transformer特征
- **易于使用**: 简单的训练和推理脚本
- **生产就绪**: 清晰、有组织的代码库,适合研究和生产

## 支持的数据集

| 数据集 | 类别数 | 任务 | 图像尺寸 |
|---------|---------|------|------------|
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 7 | 土地覆盖分类 | 可变 |
| [iSAID](https://captain-whu.github.io/iSAID/) | 16 | 航空场景理解 | 可变 |
| [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 5 | 城市语义标注 | 可变 |
| [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 | 城市语义标注 | 可变 |
| [LandCover.ai](https://landcover.ai/) | 5 | 土地覆盖分类 | 512x512 |
| [OpenEarthMap](https://open-earth-map.org/) | 8 | 全球土地覆盖制图 | 可变 |

## 项目结构

```
remote_sensing_segmentation/
├── README.md                  # 本文件
├── README_CN.md              # 中文说明
├── requirements.txt           # Python依��
├── .gitignore                # Git忽略规则
│
├── datasets/                  # 数据集加载器
│   ├── __init__.py
│   ├── loveda_dataset.py
│   ├── isaid_dataset.py
│   ├── vaihingen_dataset.py
│   ├── potsdam_dataset.py
│   ├── landcoverai_dataset.py
│   ├── openearthmap_dataset.py
│   ├── LoveDA/               # 数据集文件 (不在git中)
│   ├── iSAID/                # 数据集文件 (不在git中)
│   ├── Vaihingen/            # 数据集文件 (不在git中)
│   ├── Potsdam/              # 数据集文件 (不在git中)
│   ├── LandCoverai/          # 数据集文件 (不在git中)
│   └── OpenEarthMap/         # 数据集文件 (不在git中)
│
├── models/                    # 模型架构
│   ├── __init__.py
│   └── segmentation_model.py
│
├── scripts/                   # 训练和推理脚本
│   ├── train.py              # 主训练脚本
│   └── generate_predictions.py
│
├── tests/                     # 单元测试
│   └── test_dataset.py
│
├── docs/                      # 文档
│   ├── DATASET_SETUP.md      # 数据集设置指南 (英文)
│   ├── DATASET_SETUP_CN.md   # 数据集设置指南 (中文)
│   ├── TRAINING_RESULTS.md   # 训练结果
│   └── FINAL_COMPREHENSIVE_REPORT.md
│
└── configs/                   # 配置文件
```

## 安装

### 前置要求

- Python 3.8+
- CUDA 11.0+ (用于GPU训练)
- DINOv3仓库

### 设置步骤

1. 克隆DINOv3仓库和本项目:

```bash
# 克隆 DINOv3
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3

# 将本仓库克隆到 dinov3/ 目录下
git clone <本仓库URL> remote_sensing_segmentation
cd remote_sensing_segmentation
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 下载数据集和模型:

**由于GitHub文件大小限制，数据集和训练好的模型托管在百度网盘上。**

📦 **百度网盘下载**: https://pan.baidu.com/s/5CXLX9bODEHBSVfKVRLsmdg

网盘包含内容:
- `model/` - 6个数据集的训练模型文件（约7.2GB）
- `数据集/` - 完整数据集（可选，也可从官方源下载）

详细下载说明和文件结构请查看:
- [下载指南 (中文)](docs/LARGE_FILES_CN.md)
- [Download Guide (English)](docs/LARGE_FILES.md)

或者，您也可以自行下载原始数据集并按照 [数据集设置指南](docs/DATASET_SETUP_CN.md) 进行配置

## 快速开始

### 训练

训练单个数据集:

```bash
cd /path/to/dinov3
python remote_sensing_segmentation/scripts/train.py \
    --datasets loveda \
    --batch_size 4 \
    --num_workers 4
```

训练多个数据集:

```bash
python remote_sensing_segmentation/scripts/train.py \
    --datasets loveda isaid vaihingen \
    --batch_size 4 \
    --num_workers 4
```

训练所有数据集:

```bash
python remote_sensing_segmentation/scripts/train.py \
    --datasets all \
    --batch_size 4 \
    --num_workers 4
```

### 推理

为训练好的模型生成预测:

```bash
python remote_sensing_segmentation/scripts/generate_predictions.py \
    --datasets loveda \
    --models_dir trained_models/quick_train \
    --output_dir predictions
```

## 配置

### 训练参数

- `--datasets`: 要训练的数据集 (`all` 或具体名称)
- `--batch_size`: 训练批次大小 (默认: 4)
- `--num_workers`: 数据加载工作进程数 (默认: 4)
- `--img_size`: 训练图像尺寸 (默认: 512)
- `--backbone`: DINOv3骨干网络变体 (默认: dinov3_vitl16)

### 数据集特定设置

不同数据集可以使用不同的epoch数。详见 `scripts/train.py` 中的配置。

## 模型架构

分割模型使用:
- **骨干网络**: DINOv3 Vision Transformer (冻结或微调)
- **分割头**: 轻量级分割解码器
- **损失函数**: 交叉熵损失,可选类别权重

## 结果

训练结果和性能指标可以在以下文档中找到:
- [训练结果](docs/TRAINING_RESULTS.md)
- [综合报告](docs/FINAL_COMPREHENSIVE_REPORT.md)

## 测试

运行测试以验证数据集加载:

```bash
python tests/test_dataset.py
```

## 项目依赖

本项目依赖DINOv3仓库作为骨干模型。确保:
1. 首先克隆DINOv3
2. 将本项目放在DINOv3目录下
3. 按照DINOv3的设置说明下载预训练权重

## 许可证

本项目使用与DINOv3相同的许可证。请参考原始DINOv3仓库了解许可证详情。

## 引用

如果您在研究中使用此代码,请引用:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## 贡献

欢迎贡献! 请随时提交Pull Request。

## 联系方式

如有问题或建议,请在GitHub上开issue。

## 致谢

- Meta AI的DINOv3团队,提供了优秀的视觉Transformer骨干网络
- 数据集提供者,使其数据公开可用
- PyTorch和开源社区
