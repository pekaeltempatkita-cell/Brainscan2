# fusion_blocks.py — Feature Fusion Blocks untuk Multimodal / Multi-Scale Feature Aggregation

class FeatureFusionBlock:
    """
    Modul penggabungan representasi fitur spasial dan fitur klinis.
    """
    def __init__(self, in_features: int = 512, out_features: int = 256):
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x
