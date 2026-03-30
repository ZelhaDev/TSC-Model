"""
Traffic Sign Classifier — CNN Architecture (Improved)
======================================================
Custom CNN with residual skip connections, Global Average Pooling,
and configurable depth. Built from scratch for GTSRB (43 classes).
Satisfies the "at least one model from scratch" requirement.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two-conv residual block with BatchNorm + ReLU.
    If in_channels != out_channels, a 1x1 projection shortcut is used.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)

        # Shortcut projection when channel count changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)   # residual connection
        out = self.pool(out)
        out = self.dropout(out)
        return out


class TrafficSignCNN(nn.Module):
    """
    Configurable-depth CNN with residual blocks + Global Average Pooling.

    Args:
        num_classes: Number of output classes (43 for GTSRB).
        dropout: Dropout probability.
        num_blocks: Number of residual blocks (2, 3, or 4). Default 3.

    Input : (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    # Channel progression per block depth
    CHANNEL_CONFIGS = {
        2: [32, 64],
        3: [32, 64, 128],
        4: [32, 64, 128, 256],
    }

    def __init__(self, num_classes: int = 43, dropout: float = 0.25,
                 num_blocks: int = 3):
        super().__init__()
        assert num_blocks in self.CHANNEL_CONFIGS, \
            f"num_blocks must be one of {list(self.CHANNEL_CONFIGS.keys())}"

        channels = self.CHANNEL_CONFIGS[num_blocks]
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(ResidualBlock(in_ch, out_ch, dropout))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Global Average Pooling → FC classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def get_feature_layer(self):
        """Return the last conv layer (for Grad-CAM)."""
        # Last ResidualBlock's conv2
        return self.features[-1].conv2


# ------------------------------------------------------------------
# Quick shape verification
# ------------------------------------------------------------------
if __name__ == "__main__":
    for nb in [2, 3, 4]:
        model = TrafficSignCNN(num_classes=43, num_blocks=nb)
        dummy = torch.randn(2, 3, 32, 32)
        out = model(dummy)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{nb}-block] Input: {dummy.shape} → Output: {out.shape}  "
              f"Params: {total_params:,}")
    print("CNN shape check OK ✓")
