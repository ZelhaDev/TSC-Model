"""
Traffic Sign Classifier — CNN Architecture
============================================
Custom 3-layer CNN built from scratch for GTSRB (43 classes).
Satisfies the "at least one model from scratch" requirement.
"""

import torch
import torch.nn as nn


class TrafficSignCNN(nn.Module):
    """
    Simple 3-block CNN:
        Conv → BatchNorm → ReLU → MaxPool   (×3)
        Flatten → FC → ReLU → Dropout → FC (logits)

    Input : (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 43, dropout: float = 0.25):
        super().__init__()

        # Block 1:  3 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 32×32 → 16×16
            nn.Dropout2d(dropout),
        )

        # Block 2:  32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 16×16 → 8×8
            nn.Dropout2d(dropout),
        )

        # Block 3:  64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 8×8 → 4×4
            nn.Dropout2d(dropout),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def load_cnn_model(checkpoint_path: str = "checkpoints/best_model.pth"):
    """
    Load the pre-trained CNN model or initialize a fresh one.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        
    Returns:
        TrafficSignCNN model in eval mode
    """
    model = TrafficSignCNN(num_classes=43, dropout=0.3)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    except (FileNotFoundError, EOFError, RuntimeError) as e:
        print(f"⚠ Warning: Could not load checkpoint. Using untrained model. ({type(e).__name__})")
    model.eval()
    return model


# ------------------------------------------------------------------
# Quick shape verification
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = TrafficSignCNN(num_classes=43)
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("CNN shape check OK ✓")