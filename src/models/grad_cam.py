"""
Traffic Sign Classifier — Grad-CAM Explainability
====================================================
Generates Grad-CAM saliency overlays for the trained CNN.
Visualizes where the model focuses for correct and incorrect predictions.

Run:
    python src/grad_cam.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data.data_pipeline import load_config, get_dataloaders
from src.models.cnn import TrafficSignCNN


# GTSRB class names
GTSRB_NAMES = [
    "20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h",
    "80 km/h", "End 80", "100 km/h", "120 km/h", "No passing",
    "No pass >3.5t", "Priority", "Priority road", "Yield", "Stop",
    "No vehicles", "No >3.5t", "No entry", "General caution",
    "Left curve", "Right curve", "Double curve", "Bumpy road",
    "Slippery", "Narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children", "Bicycles", "Ice/snow",
    "Wild animals", "End limits", "Right turn", "Left turn",
    "Ahead only", "Ahead/right", "Ahead/left", "Keep right",
    "Keep left", "Roundabout", "End no passing", "End no pass >3.5t",
]


class GradCAM:
    """
    Grad-CAM implementation for TrafficSignCNN.
    Captures gradients at the target convolutional layer
    and produces class-discriminative saliency maps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, 3, H, W).
        target_class : int or None
            If None, uses predicted class.

        Returns
        -------
        heatmap : np.ndarray
            Normalized heatmap (H, W) in [0, 1].
        predicted_class : int
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input dimensions
        cam_resized = np.array(
            F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0).float(),
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear", align_corners=False,
            ).squeeze().numpy()
        )

        return cam_resized, target_class


def overlay_heatmap(img_np, heatmap, alpha=0.4):
    """Overlay heatmap on image with jet colormap."""
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3) RGB
    overlay = (1 - alpha) * img_np + alpha * colormap
    return np.clip(overlay, 0, 1)


def denormalize(img_tensor):
    """De-normalize image tensor for display."""
    mean = np.array([0.3401, 0.3120, 0.3212]).reshape(3, 1, 1)
    std = np.array([0.2725, 0.2609, 0.2669]).reshape(3, 1, 1)
    img = img_tensor.numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def generate_grad_cam_gallery(model, test_loader, device, save_path,
                              num_correct=8, num_wrong=8):
    """Generate a Grad-CAM gallery with correct and incorrect predictions."""
    target_layer = model.get_feature_layer()
    grad_cam = GradCAM(model, target_layer)

    correct_samples = []
    wrong_samples = []

    for imgs, labels in test_loader:
        for i in range(imgs.size(0)):
            if len(correct_samples) >= num_correct and \
               len(wrong_samples) >= num_wrong:
                break

            img = imgs[i:i+1].to(device)
            true_label = labels[i].item()

            with torch.enable_grad():
                img_grad = img.clone().requires_grad_(False)
                heatmap, pred_class = grad_cam.generate(img_grad, target_class=None)

            is_correct = (pred_class == true_label)

            entry = {
                "image": imgs[i],
                "heatmap": heatmap,
                "true": true_label,
                "pred": pred_class,
                "correct": is_correct,
            }

            if is_correct and len(correct_samples) < num_correct:
                correct_samples.append(entry)
            elif not is_correct and len(wrong_samples) < num_wrong:
                wrong_samples.append(entry)

        if len(correct_samples) >= num_correct and \
           len(wrong_samples) >= num_wrong:
            break

    # Plot gallery
    all_samples = correct_samples + wrong_samples
    n = len(all_samples)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(all_samples):
        row = i // cols
        col = (i % cols) * 2

        img_np = denormalize(sample["image"])
        heatmap = sample["heatmap"]
        overlay = overlay_heatmap(img_np, heatmap)

        # Original image
        axes[row, col].imshow(img_np)
        color = "green" if sample["correct"] else "red"
        axes[row, col].set_title(
            f"True: {GTSRB_NAMES[sample['true']]}\n"
            f"Pred: {GTSRB_NAMES[sample['pred']]}",
            fontsize=7, color=color
        )
        axes[row, col].axis("off")

        # Grad-CAM overlay
        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].set_title("Grad-CAM", fontsize=7)
        axes[row, col + 1].axis("off")

    # Hide unused axes
    for i in range(len(all_samples), rows * cols):
        row = i // cols
        col = (i % cols) * 2
        if row < axes.shape[0] and col + 1 < axes.shape[1]:
            axes[row, col].axis("off")
            axes[row, col + 1].axis("off")

    plt.suptitle(
        f"Grad-CAM Visualizations — "
        f"Correct (green, top) & Misclassified (red, bottom)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM gallery saved → {save_path}")


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    print("Loading data ...")
    _, _, test_loader, num_classes = get_dataloaders(cfg)

    # Load best model
    ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    model = TrafficSignCNN(num_classes=num_classes,
                           dropout=cfg["cnn"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    print(f"Loaded model from {ckpt}")

    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, "grad_cam_examples.png")
    generate_grad_cam_gallery(model, test_loader, device, save_path)

    print("\n✓ Grad-CAM generation complete.")


if __name__ == "__main__":
    main()
