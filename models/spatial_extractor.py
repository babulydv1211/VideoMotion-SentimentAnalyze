"""
Spatial Feature Extractor
Extract spatial features using ResNet18
"""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class SpatialFeatureExtractor(nn.Module):

    def __init__(
        self,
        pretrained=True,
        feature_dim=512
    ):

        super().__init__()

        # ============================================
        # LOAD RESNET18
        # ============================================

        if pretrained:

            backbone = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

        else:

            backbone = resnet18(
                weights=None
            )

        # ============================================
        # REMOVE FINAL CLASSIFICATION LAYER
        # ============================================

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = feature_dim

        # ============================================
        # ENABLE TRAINING
        # ============================================

        for param in self.backbone.parameters():

            param.requires_grad = True

    def forward(self, frames):

        """
        Input:
            (B, T, 3, H, W)
            OR
            (B, 3, H, W)

        Output:
            (B, T, 512)
            OR
            (B, 512)
        """

        # ============================================
        # VIDEO INPUT
        # ============================================

        if len(frames.shape) == 5:

            B, T, C, H, W = frames.shape

            frames = frames.reshape(
                B * T,
                C,
                H,
                W
            )

            features = self.backbone(frames)

            # (B*T, 512, 1, 1)

            features = features.view(
                B * T,
                self.feature_dim
            )

            features = features.view(
                B,
                T,
                self.feature_dim
            )

        # ============================================
        # SINGLE IMAGE
        # ============================================

        else:

            features = self.backbone(frames)

            features = features.view(
                features.size(0),
                self.feature_dim
            )

        return features

    def freeze_backbone(self):

        for param in self.backbone.parameters():

            param.requires_grad = False

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():

            param.requires_grad = True