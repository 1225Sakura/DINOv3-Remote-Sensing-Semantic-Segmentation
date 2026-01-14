# Copyright (c) Remote Sensing Segmentation Training
# Semantic Segmentation Model with DINOv3 Backbone

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head for semantic segmentation.
    Takes multi-scale features from backbone and produces segmentation logits.
    """

    def __init__(self, in_channels, num_classes, feature_scale=4, dropout=0.1):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_classes: Number of segmentation classes
            feature_scale: Scale of features (default 4 means features are 1/4 of input size)
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_scale = feature_scale

        # Multi-scale feature fusion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, target_size=None):
        """
        Args:
            features: Features from backbone (B, C, H, W)
            target_size: Target output size (H, W), if None uses 4x feature size

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.classifier(x)

        # Upsample to target size
        if target_size is None:
            target_size = (x.size(2) * self.feature_scale, x.size(3) * self.feature_scale)

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class DINOv3SegmentationModel(nn.Module):
    """
    Complete segmentation model with DINOv3 backbone and segmentation head.
    """

    def __init__(self, backbone, num_classes, freeze_backbone=True, dropout=0.1):
        """
        Args:
            backbone: DINOv3 vision transformer backbone
            num_classes: Number of segmentation classes
            freeze_backbone: Whether to freeze backbone weights
            dropout: Dropout rate in segmentation head
        """
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")

        # Get embedding dimension from backbone
        embed_dim = backbone.embed_dim
        patch_size = backbone.patch_size

        print(f"Backbone embed_dim: {embed_dim}, patch_size: {patch_size}")

        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=embed_dim,
            num_classes=num_classes,
            feature_scale=patch_size,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Get features from backbone
        # DINOv3 forward_features returns (B, N, C) where N = H/patch_size * W/patch_size
        features = self.backbone.forward_features(x)

        # Remove CLS token and reshape to spatial
        # features shape: (B, 1 + H*W/patch_size^2, C)
        # We need to remove the CLS token (first token)
        B, N, C = features['x_norm_patchtokens'].shape
        H_feat = W_feat = int(N ** 0.5)

        # Reshape to (B, C, H, W)
        features_spatial = features['x_norm_patchtokens'].transpose(1, 2).reshape(B, C, H_feat, W_feat)

        # Get segmentation output
        output = self.seg_head(features_spatial, target_size=(x.size(2), x.size(3)))

        return output

    def predict(self, x):
        """
        Prediction with argmax.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Predicted class IDs (B, H, W)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class UPerNetHead(nn.Module):
    """
    UPerNet-style segmentation head with FPN structure.
    More advanced than simple SegmentationHead.
    """

    def __init__(self, in_channels, num_classes, channels=512, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        # PPM (Pyramid Pooling Module)
        self.ppm = PyramidPoolingModule(in_channels, channels // 4, bins=(1, 2, 3, 6))

        # FPN convs
        self.fpn_in = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.fpn_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(channels + channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def forward(self, features, target_size):
        # PPM
        ppm_out = self.ppm(features)

        # FPN
        fpn_feature = self.fpn_in(features)
        fpn_feature = self.fpn_out(fpn_feature)

        # Concatenate
        output = torch.cat([fpn_feature, ppm_out], dim=1)

        # Classify
        output = self.classifier(output)

        # Upsample
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)

        return output


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module"""

    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for bin in bins
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(bins) * out_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        h, w = features.size()[2:]
        pyramids = [features]

        for stage in self.stages:
            pyramids.append(F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=False))

        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


if __name__ == "__main__":
    # Test the model
    print("Testing segmentation model...")

    # Create a dummy backbone
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 1024
            self.patch_size = 16
            self.conv = nn.Conv2d(3, 1024, kernel_size=16, stride=16)

        def forward_features(self, x):
            B, C, H, W = x.shape
            features = self.conv(x)
            B, C, H_f, W_f = features.shape
            features = features.flatten(2).transpose(1, 2)
            return {'x_norm_patchtokens': features}

        def parameters(self):
            return self.conv.parameters()

    backbone = DummyBackbone()
    model = DINOv3SegmentationModel(backbone, num_classes=7, freeze_backbone=False)

    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
