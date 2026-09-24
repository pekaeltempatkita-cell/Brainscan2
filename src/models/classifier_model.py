"""
classifier_model.py — Model utama HybridViTEfficientNet (EfficientNet-B3 + ViT + Fusion).
HARUS identik dengan training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

from models.fusion_blocks import PatchEmbedding, TransformerBlock, CrossModalAttentionFusion
from config import IMG_SIZE


class HybridViTEfficientNet(nn.Module):
    def __init__(self, num_classes=10, vit_embed_dim=768, vit_num_heads=12,
                 vit_num_layers=6, fusion_dim=512, dropout=0.3):
        super().__init__()
        backbone = efficientnet_b3(weights=None)
        cnn_out = 1536
        self.features = backbone.features

        self.patch_embed = PatchEmbedding(cnn_out, patch_size=1, embed_dim=vit_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            dummy_feat = self.features(dummy)
            num_patches = dummy_feat.shape[2] * dummy_feat.shape[3]

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(vit_embed_dim, vit_num_heads, dropout=dropout)
            for _ in range(vit_num_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_embed_dim)

        self.fusion = CrossModalAttentionFusion(cnn_dim=cnn_out, vit_dim=vit_embed_dim,
                                                 fusion_dim=fusion_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat_map = self.features(x)
        cnn_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        patches = self.patch_embed(feat_map)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.vit_norm(tokens)
        vit_feat = tokens[:, 0]
        fused = self.fusion(cnn_feat, vit_feat)
        return self.classifier(fused)