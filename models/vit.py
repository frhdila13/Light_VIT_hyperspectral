import torch
import torch.nn as nn
from vit_pytorch import ViT

class vit(nn.Module):
    def __init__(self, num_classes, num_bands, patch_size=8):
        super().__init__()
        
        # It expects (Batch, Channels, Height, Width)
        self.model = ViT(
            image_size = patch_size,
            patch_size = 2,          # Must be a divisor of your patch_size (8)
            num_classes = num_classes,
            channels = num_bands,    # 128 for Chikusei
            dim = 1024,               # Hidden dimension
            depth = 6,               # Number of layers
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        # Your data comes in as (B, 1, C, H, W) from the dataset.py
        # SimpleViT needs (B, C, H, W)
        if x.dim() == 5:
            x = x.squeeze(1)
        
        return self.model(x)
