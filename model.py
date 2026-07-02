"""
model.py

Embedding CNN for fingerprint re-identification.

Unlike a classifier, this network does NOT output class scores. It maps an
input fingerprint image to a fixed-length vector (an "embedding") such that
images of the SAME finger end up close together in that vector space, and
images of DIFFERENT fingers end up far apart -- regardless of whether that
finger's identity was seen during training. That property (generalizing to
unseen identities) is what makes this approach usable for open-set
identification, where new subjects can be enrolled without retraining the
network, unlike a fixed 600-way softmax head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FingerprintEmbeddingNet(nn.Module):
    """Small ResNet-ish CNN, sized for 96x96 grayscale fingerprint crops.

    Architecture rationale:
      - Strided convs instead of pooling for downsampling: preserves more
        ridge-orientation information than max-pooling, which can be
        important for fine ridge/minutiae detail.
      - Global average pooling before the embedding head: makes the network
        robust to the small translation/rotation offsets that occur between
        different scans of the same finger.
      - Final L2 normalization: makes embedding distance behave like cosine
        similarity, which is what triplet loss / verification thresholds
        assume, and keeps embeddings on a fixed-scale hypersphere so
        distances are comparable across images.
    """

    def __init__(self, embedding_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.stem = ConvBlock(in_channels, 32, stride=1)

        self.stage1 = nn.Sequential(ConvBlock(32, 32), ConvBlock(32, 64, stride=2))
        self.stage2 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 128, stride=2))
        self.stage3 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 256, stride=2))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # unit-length embeddings
        return x


if __name__ == "__main__":
    net = FingerprintEmbeddingNet(embedding_dim=128)
    dummy = torch.randn(4, 1, 96, 96)
    out = net(dummy)
    print("Output shape:", out.shape)
    print("Norms (should all be ~1.0):", out.norm(dim=1))
    print("Total params:", sum(p.numel() for p in net.parameters()))
