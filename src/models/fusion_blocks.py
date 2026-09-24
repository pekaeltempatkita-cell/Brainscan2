"""
fusion_blocks.py — Komponen-komponen kecil pembentuk arsitektur ViT & fusion.
HARUS identik dengan yang dipakai saat training -- ini bukan tempat improvisasi.
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Ubah feature map CNN jadi 'token' yang bisa diproses Transformer."""
    def __init__(self, in_channels=1536, patch_size=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)   # (B, C, H, W) -> (B, N_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalAttentionFusion(nn.Module):
    """Adaptive Attention Gate -- menentukan bobot kontribusi fitur CNN vs ViT secara dinamis."""
    def __init__(self, cnn_dim=1536, vit_dim=768, fusion_dim=512, dropout=0.3):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, cnn_feat, vit_feat):
        c = self.cnn_proj(cnn_feat)
        v = self.vit_proj(vit_feat)
        w = self.attn(torch.cat([c, v], dim=-1))
        fused = w[:, 0:1] * c + w[:, 1:2] * v
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused