import torch
import torch.nn as nn
from torchvision.models.swin_transformer import (
    SwinTransformer,
    SwinTransformerBlock,
    PatchMerging
)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class SwinTransformerBackbone(nn.Module):
    embed_dim = 96
    patch_size = [4, 4]
    depths = [2, 6, 2]
    num_heads = [6, 12, 24]
    window_size = [8, 8]
    dropout = 0.1

    def __init__(self, num_classes=36):
        super().__init__()
        self.model = SwinTransformer(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            dropout=self.dropout,
            block=SwinTransformerBlock,
            downsample_layer=PatchMerging,
            num_classes=num_classes
        )
        self.model.features[0] = nn.Sequential(
            nn.Conv2d(
                1, self.embed_dim,
                kernel_size=(self.patch_size[0], self.patch_size[1]),
                stride=(self.patch_size[0], self.patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn(4, 1, 112, 112)
    model = SwinTransformerBackbone()
    print(model(x).shape)
    # print total params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
