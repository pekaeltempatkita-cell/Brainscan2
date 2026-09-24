"""
precheck_model.py — Model precheck (Brain vs Non_Brain gate).
Arsitektur: EfficientNet-B0. HARUS identik dengan training.
"""
import torch.nn as nn
from torchvision.models import efficientnet_b0


class PrecheckModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = efficientnet_b0(weights=None).features  # ditimpa checkpoint, gak perlu ImageNet
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))