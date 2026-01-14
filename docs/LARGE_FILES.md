# 大文件下载说明

本项目包含一些大文件（模型权重、日志文件等），这些文件已被`.gitignore`排除，不会上传到GitHub。您可以从以下网盘下载：

## 百度网盘下载链接

**Share Name**: dinov3-遥感语义 (DINOv3 Remote Sensing Semantic Segmentation)

**Download Link**: https://pan.baidu.com/s/5CXLX9bODEHBSVfKVRLsmdg

**Directory Structure in Cloud**:
```
dinov3-遥感语义/
├── model/           # Trained model files (~7.2GB)
│   └── trained_models/
│       ├── isaid/
│       ├── landcoverai/
│       ├── loveda/
│       ├── openearthmap/
│       ├── potsdam/
│       └── vaihingen/
└── 数据集/          # 6 Remote Sensing Datasets (optional)
    ├── LoveDA/
    ├── iSAID/
    ├── Vaihingen/
    ├── Potsdam/
    ├── LandCoverai/
    └── OpenEarthMap/
```

> **Note**:
> - For inference only: Download the `model` folder
> - For training or complete datasets: Download the `数据集` (Datasets) folder

---

## 文件清单

### 1. 训练好的模型文件 (共 ~7.2 GB)

所有模型文件位于 `trained_models/` 目录下，每个数据集一个子目录：

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `trained_models/isaid/model.pth` | 1.2 GB | iSAID数据集训练的模型 |
| `trained_models/landcoverai/model.pth` | 1.2 GB | LandCover.ai数据集训练的模型 |
| `trained_models/loveda/model.pth` | 1.2 GB | LoveDA数据集训练的模型 |
| `trained_models/openearthmap/model.pth` | 1.2 GB | OpenEarthMap数据集训练的模型 |
| `trained_models/potsdam/model.pth` | 1.2 GB | Potsdam数据集训练的模型 |
| `trained_models/vaihingen/model.pth` | 1.2 GB | Vaihingen数据集训练的模型 |

每个模型目录还包含一个 `results.json` 文件（几KB），记录训练结果和指标。

### 2. 训练日志文件 (共 ~229 MB)

日志文件位于 `logs/` 目录：

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `logs/training_output.log` | 97 MB | 主训练输出日志 |
| `logs/full_training_fixed.log` | 54 MB | 完整训练日志（修正版） |
| `logs/full_training.log` | 47 MB | 完整训练日志 |
| `logs/training_output_previous.log` | 31 MB | 之前的训练输出日志 |

### 3. 预训练权重（可选）

如需使用DINOv3预训练权重，请从以下来源下载：

- **官方下载**: [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- 推荐模型: `dinov3_vitl16_pretrain_lvd1689m.pth`
- 下载后放置在项目根目录的 `checkpoints/` 文件夹

### 4. 数据集（需单独下载）

训练使用的数据集不包含在此分享中，请从官方网站下载：

| 数据集 | 官方链接 | 放置路径 |
|--------|---------|----------|
| LoveDA | [链接](https://github.com/Junjue-Wang/LoveDA) | `datasets/LoveDA/` |
| iSAID | [链接](https://captain-whu.github.io/iSAID/) | `datasets/iSAID/` |
| Vaihingen | [链接](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | `datasets/Vaihingen/` |
| Potsdam | [链接](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | `datasets/Potsdam/` |
| LandCover.ai | [链接](https://landcover.ai/) | `datasets/LandCoverai/` |
| OpenEarthMap | [链接](https://open-earth-map.org/) | `datasets/OpenEarthMap/` |

详细的数据集下载和配置说明请参考 [Dataset Setup Guide](DATASET_SETUP.md)

---

## 使用说明

### 下载后的文件放置

1. **模型文件**: 解压后保持目录结构，放在项目根目录
   ```
   remote_sensing_segmentation_project/
   └── trained_models/
       ├── isaid/
       │   ├── model.pth
       │   └── results.json
       ├── landcoverai/
       │   ├── model.pth
       │   └── results.json
       └── ...
   ```

2. **日志文件**: 放在 `logs/` 目录（可选，仅供参考）
   ```
   remote_sensing_segmentation_project/
   └── logs/
       ├── training_output.log
       ├── full_training.log
       └── ...
   ```

3. **预训练权重**: 创建 `checkpoints/` 目录并放入
   ```
   remote_sensing_segmentation_project/
   └── checkpoints/
       └── dinov3_vitl16_pretrain.pth
   ```

### 验证下载完整性

下载完成后，您可以运行以下命令验证文件：

```bash
# 检查模型文件
ls -lh trained_models/*/model.pth

# 查看训练结果
cat trained_models/*/results.json
```

---

## 文件用途

- **模型文件 (.pth)**: 已训练好的模型权重，可直接用于推理预测
- **日志文件 (.log)**: 训练过程的详细日志，记录了训练损失、验证指标等
- **结果文件 (.json)**: 各数据集的训练结果汇总，包含mIoU、准确率等指标

---

## 常见问题

**Q: 我只想测试某个数据集，需要下载所有模型吗？**
A: 不需要，只下载对应数据集的模型文件即可。例如只测试LoveDA，下载 `trained_models/loveda/` 目录即可。

**Q: 日志文件必须下载吗？**
A: 不必须。日志文件仅供参考训练过程，不影响模型使用。

**Q: 如何使用下载的模型进行预测？**
A: 参考主README中的使用说明，或运行：
```bash
python scripts/generate_predictions.py --datasets loveda --models_dir trained_models
```

---

## 联系方式

如果下载链接失效或有其他问题，请提交Issue。
