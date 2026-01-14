# 数据集设置指南

本文档提供了在本项目中使用的六个遥感语义分割数据集的详细设置说明。

## 支持的数据集

1. **LoveDA** - 用于领域自适应语义分割的土地覆盖数据集
2. **iSAID** - 航空图像中的实例分割数据集
3. **Vaihingen** - ISPRS 2D语义标注竞赛
4. **Potsdam** - ISPRS 2D语义标注竞赛
5. **LandCover.ai** - 高分辨率土地覆盖分类
6. **OpenEarthMap** - 全球土地覆盖制图数据集

## 数据集结构

所有数据集应放置在 `datasets/` 目录下,结构如下:

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

## 1. LoveDA 数据集

**下载:** https://github.com/Junjue-Wang/LoveDA

**期望结构:**
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

**类别:** 7个类别 (背景、建筑、道路、水体、裸地、森林、农田)

## 2. iSAID 数据集

**下载:** https://captain-whu.github.io/iSAID/

**期望结构:**
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

**类别:** 16个类别 (背景、船舶、储罐、棒球场、网球场、篮球场、田径场、桥梁、大型车辆、小型车辆、直升机、游泳池、环岛、足球场、飞机、港口)

**注意事项:**
- 标注是RGB编码的图像
- 原始图像应放在 `images/` 文件夹中
- 语义标注应放在 `Semantic_masks/images/` 文件夹中

## 3. Vaihingen 数据集

**下载:** https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx

**期望结构:**
```
Vaihingen/
├── train/
│   ├── train_image_rgb/
│   └── train_label_rgb/
└── test/
    ├── test_image_rgb/
    └── test_label_rgb/
```

**类别:** 5个类别 (不透水表面、建筑、低矮植被、树木、汽车)

**注意事项:**
- 图像可以是 .tif 或 .png 格式
- 标签是RGB编码的

## 4. Potsdam 数据集

**下载:** https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

**期望结构:**
```
Potsdam/
├── train/
│   ├── train_image_rgb/
│   └── train_label_rgb/
└── test/
    ├── test_image_rgb/
    └── test_label_rgb/
```

**类别:** 6个类别 (不透水表面、建筑、低矮植被、树木、汽车、杂物/背景)

## 5. LandCover.ai 数据集

**下载:** https://landcover.ai/

**期望结构:**
```
LandCoverai/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

**类别:** 5个类别 (背景、建筑物、林地、水体、道路)

## 6. OpenEarthMap 数据集

**下载:** https://open-earth-map.org/

**期望结构:**
```
OpenEarthMap/
├── Train/
│   ├── images/
│   └── labels/
└── Val/
    ├── images/
    └── labels/
```

**类别:** 8个类别 (裸地、草地、路面、道路、树木、水体、农田、建筑)

**注意事项:**
- 图像为TIFF格式
- 使用非连续标签值 (0,2,3,4,5,6,7,8),会自动映射为连续值 (0-7)
- 需要 `tifffile` 库来读取TIFF图像

## 设置说明

1. 创建数据集目录:
   ```bash
   mkdir -p datasets
   ```

2. 从上述链接下载各个数据集

3. 按照显示的结构解压和组织它们

4. 验证数据集加载:
   ```bash
   python tests/test_dataset.py
   ```

## 注意事项

- 所有数据集使用不同的图像尺寸和格式
- 训练期间图像会自动调整为512x512
- 训练时应用数据增强以提高泛化能力
- RGB编码的标注会自动转换为类别ID
