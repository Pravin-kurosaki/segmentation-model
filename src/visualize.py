"""
visualize.py - Visualization utilities for segmentation predictions.

Generates side-by-side panels:
  Original Image | Ground Truth Mask | Predicted Mask | Overlay

Saves PNG files to results/visualizations/.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows/servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_color(mask: np.ndarray, class_colors: dict) -> np.ndarray:
    """
    Convert a single-channel class-ID mask to an RGB colour image.

    Args:
        mask         : (H, W) int array of class IDs
        class_colors : {class_id: (R, G, B)} mapping

    Returns:
        (H, W, 3) uint8 RGB array
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in class_colors.items():
        color_img[mask == cls_id] = rgb
    return color_img


def overlay_mask(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Alpha-blend a colour mask over an RGB image.

    Args:
        image      : (H, W, 3) uint8 RGB image
        color_mask : (H, W, 3) uint8 colour mask
        alpha      : mask opacity (0=transparent, 1=opaque mask)

    Returns:
        (H, W, 3) uint8 blended image
    """
    blended = (image.astype(np.float32) * (1 - alpha)
               + color_mask.astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert tensor to uint8 numpy array.

    Args:
        tensor : (3, H, W) float32 normalised tensor

    Returns:
        (H, W, 3) uint8 numpy array
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img  = img * std + mean
    img  = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def build_legend(class_colors: dict, class_names: dict, present_classes: set):
    """Build matplotlib legend patches for classes present in the image."""
    patches = []
    for cls_id in sorted(present_classes):
        name = class_names.get(cls_id, f"class_{cls_id}")
        rgb  = class_colors.get(cls_id, (128, 128, 128))
        patches.append(
            mpatches.Patch(color=[c / 255.0 for c in rgb], label=name)
        )
    return patches


# ---------------------------------------------------------------------------
# Core visualisation function
# ---------------------------------------------------------------------------

def visualize_sample(
    image_tensor : torch.Tensor,
    gt_mask      : np.ndarray,
    pred_mask    : np.ndarray,
    stem         : str,
    save_path    : Path,
    class_colors : dict = None,
    class_names  : dict = None,
    alpha        : float = 0.45,
) -> None:
    """
    Create and save a 4-panel visualisation figure.

    Panels: Original | Ground Truth | Prediction | Overlay (prediction blended on image)

    Args:
        image_tensor : (3, H, W) normalised float32 tensor
        gt_mask      : (H, W) int array -- ground truth class IDs
        pred_mask    : (H, W) int array -- predicted class IDs
        stem         : image ID string (used in title)
        save_path    : full path to save the PNG
        class_colors : {class_id: (R, G, B)} - defaults to cfg.CLASS_COLORS
        class_names  : {class_id: name}       - defaults to cfg.ACTIVE_CLASSES
        alpha        : overlay transparency
    """
    if class_colors is None:
        class_colors = cfg.CLASS_COLORS
    if class_names is None:
        class_names  = cfg.ACTIVE_CLASSES

    img_np   = denormalize(image_tensor)          # (H, W, 3) uint8
    gt_color = mask_to_color(gt_mask,   class_colors)
    pr_color = mask_to_color(pred_mask, class_colors)
    overlay  = overlay_mask(img_np, pr_color, alpha)

    present = set(gt_mask.flatten().tolist()) | set(pred_mask.flatten().tolist())
    patches = build_legend(class_colors, class_names, present)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Image: {stem}", fontsize=13, fontweight="bold")

    panels = [
        (img_np,   "Original Image"),
        (gt_color, "Ground Truth"),
        (pr_color, "Prediction"),
        (overlay,  "Overlay (pred)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if patches:
        fig.legend(
            handles      = patches,
            loc          = "lower center",
            ncol         = min(len(patches), 7),
            fontsize     = 8,
            bbox_to_anchor = (0.5, -0.06),
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch visualisation (called after evaluation)
# ---------------------------------------------------------------------------

def visualize_predictions(
    model,
    loader,
    device       : torch.device,
    save_dir     : Path,
    n_samples    : int = 16,
    class_colors : dict = None,
    class_names  : dict = None,
    prefix       : str  = "pred",
) -> list:
    """
    Run inference on up to n_samples batches from loader and save visualisations.

    Args:
        model       : trained segmentation model (returns logits)
        loader      : DataLoader (test or val)
        device      : torch.device
        save_dir    : directory to save PNG files
        n_samples   : max number of images to visualise
        class_colors: colour map
        class_names : class name map
        prefix      : filename prefix

    Returns:
        List of saved file paths.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device, non_blocking=True)
            masks   = batch["mask"].numpy()          # (B, H, W) int
            stems   = batch["stem"]

            logits  = model(images)                  # (B, C, H, W)
            preds   = torch.argmax(logits, dim=1).cpu().numpy()  # (B, H, W)

            for i in range(len(stems)):
                if count >= n_samples:
                    break
                save_path = save_dir / f"{prefix}_{count:04d}_{stems[i]}.png"
                visualize_sample(
                    image_tensor = images[i].cpu(),
                    gt_mask      = masks[i],
                    pred_mask    = preds[i],
                    stem         = stems[i],
                    save_path    = save_path,
                    class_colors = class_colors,
                    class_names  = class_names,
                )
                saved_paths.append(save_path)
                count += 1

            if count >= n_samples:
                break

    print(f"  Saved {len(saved_paths)} visualisations to: {save_dir}")
    return saved_paths


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def plot_training_history(history_csv: Path, save_dir: Path) -> Path:
    """
    Plot training loss, val loss, val mIoU, and val Dice from training_history.csv.

    Args:
        history_csv : Path to training_history.csv
        save_dir    : Directory to save the plot

    Returns:
        Path to saved plot.
    """
    import pandas as pd

    df = pd.read_csv(history_csv)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#e05c5c")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#5c9ee0")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mIoU + Dice
    axes[1].plot(df["epoch"], df["val_miou"],  label="Val mIoU",  color="#5cb85c")
    axes[1].plot(df["epoch"], df["val_dice"],  label="Val Dice",  color="#f0a830")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = save_dir / "training_history.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training history plot saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  visualize.py smoke test")
    print("=" * 60)

    dummy_image = torch.zeros(3, 256, 256)   # all-black normalised tensor
    rng = np.random.default_rng(42)
    gt_mask   = rng.integers(0, cfg.NUM_CLASSES, size=(256, 256)).astype(np.int64)
    pred_mask = gt_mask.copy()
    pred_mask[::4, ::4] = rng.integers(0, cfg.NUM_CLASSES, size=(64, 64))

    out_path = cfg.VIZ_DIR / "smoke_test_visualize.png"
    visualize_sample(
        image_tensor = dummy_image,
        gt_mask      = gt_mask,
        pred_mask    = pred_mask,
        stem         = "smoke_test",
        save_path    = out_path,
    )
    print(f"  Saved: {out_path}")
    assert out_path.exists(), "File was not created!"
    print("  visualize.py smoke test PASSED")
