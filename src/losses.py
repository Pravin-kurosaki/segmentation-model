"""
losses.py — Combined loss functions for face segmentation.

Phase 10 implementation:
  - CrossEntropyLoss  : standard pixel-wise classification loss
  - DiceLoss          : overlap-based loss, effective for class imbalance
  - CombinedLoss      : weighted sum  total = w_ce * CE + w_dice * Dice

Why both?
  Cross-entropy treats every pixel independently. Dice optimises the
  overlap ratio directly, which helps small classes (nose, eyes, glasses)
  not get ignored by the model focusing only on large classes (skin, hair).

Design rules:
  - All losses accept RAW LOGITS (not softmax probabilities).
  - All losses accept integer target masks of shape (B, H, W).
  - Losses are modular — swap or add new ones in CombinedLoss easily.

Future candidates (drop-in replacements):
  - Focal loss (for extreme class imbalance)
  - Weighted cross-entropy (per-class frequency weights)
  - Lovasz-softmax (direct mIoU optimisation)
  - Tversky loss (asymmetric Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Computes Dice per class, then averages (macro mean).
    Uses soft Dice on softmax probabilities so it is fully differentiable.

    Args:
        smooth:         Laplace smoothing to avoid division by zero.
        ignore_index:   Class index to exclude from loss (-1 to disable).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W)  — raw model output
            targets : (B, H, W)     — integer class IDs
        Returns:
            scalar Dice loss (1 - mean Dice coefficient)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(
            targets.long().clamp(0, num_classes - 1),
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # Optionally zero-out the ignored class
        if self.ignore_index >= 0 and self.ignore_index < num_classes:
            targets_one_hot[:, self.ignore_index] = 0
            probs_used = probs.clone()
            probs_used[:, self.ignore_index] = 0
        else:
            probs_used = probs

        # Flatten spatial dims: (B, C, H*W)
        probs_flat   = probs_used.view(probs_used.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        # Compute Dice per class per batch item
        intersection = (probs_flat * targets_flat).sum(dim=2)    # (B, C)
        union        = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()   # scalar

        return dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss (thin wrapper for consistency)
# ─────────────────────────────────────────────────────────────────────────────

class CrossEntropyLoss(nn.Module):
    """
    Standard pixel-wise cross-entropy loss.

    Args:
        weight:        Per-class weight tensor (C,) for class imbalance.
                       Pass None to use uniform weights.
        ignore_index:  Pixels with this label are ignored (-100 to disable).
        label_smoothing: Smoothing factor (0 = standard CE, 0.1 = mild smoothing).
    """

    def __init__(
        self,
        weight:          torch.Tensor = None,
        ignore_index:    int          = -100,
        label_smoothing: float        = 0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight          = weight,
            ignore_index    = ignore_index,
            label_smoothing = label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W)  — raw model output
            targets : (B, H, W)     — integer class IDs
        Returns:
            scalar CE loss
        """
        return self.ce(logits, targets.long())


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted combination of Cross-Entropy and Dice loss.

        total_loss = ce_weight * CE(logits, targets)
                   + dice_weight * Dice(logits, targets)

    Args:
        ce_weight:       Weight for cross-entropy term (default 1.0).
        dice_weight:     Weight for Dice term (default 1.0).
        class_weights:   Per-class CE weights tensor (C,) — for imbalanced classes.
        smooth:          Dice smoothing constant.
        ignore_index:    Class index to ignore in CE loss.
        label_smoothing: CE label smoothing factor.
    """

    def __init__(
        self,
        ce_weight:       float         = 1.0,
        dice_weight:     float         = 1.0,
        class_weights:   torch.Tensor  = None,
        smooth:          float         = 1.0,
        ignore_index:    int           = -100,
        label_smoothing: float         = 0.0,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss   = CrossEntropyLoss(
            weight          = class_weights,
            ignore_index    = ignore_index,
            label_smoothing = label_smoothing,
        )
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Args:
            logits  : (B, C, H, W) — raw logits
            targets : (B, H, W)    — integer class IDs
        Returns:
            (total_loss, ce_loss, dice_loss) — all scalars
            Returns individual losses so they can be logged separately.
        """
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, ce, dice


# ─────────────────────────────────────────────────────────────────────────────
# Class-frequency weights helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    pixel_counts: dict,
    num_classes:  int,
    method:       str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class loss weights from pixel frequency counts.

    Args:
        pixel_counts: {class_id: pixel_count} from dataset inspection.
        num_classes:  Total number of classes.
        method:       'inverse_freq'  — weight = 1 / frequency (normalised)
                      'median_freq'   — weight = median_freq / class_freq

    Returns:
        Tensor of shape (num_classes,) with float weights.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for cls_id, cnt in pixel_counts.items():
        if 0 <= int(cls_id) < num_classes:
            counts[int(cls_id)] = float(cnt)

    # Avoid division by zero for classes with no pixels
    counts = counts.clamp(min=1.0)
    freq   = counts / counts.sum()

    if method == "inverse_freq":
        weights = 1.0 / freq
    elif method == "median_freq":
        median  = freq.median()
        weights = median / freq
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse_freq' or 'median_freq'.")

    # Normalise so weights average to 1
    weights = weights / weights.mean()
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run as script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import config as cfg

    B, C, H, W = 2, cfg.NUM_CLASSES, 256, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random logits and integer targets
    logits  = torch.randn(B, C, H, W).to(device)
    targets = torch.randint(0, C, (B, H, W)).to(device)

    # Test individual losses
    ce_fn   = CrossEntropyLoss()
    dice_fn = DiceLoss()

    ce_val   = ce_fn(logits, targets)
    dice_val = dice_fn(logits, targets)
    print(f"CrossEntropyLoss : {ce_val.item():.4f}")
    print(f"DiceLoss         : {dice_val.item():.4f}")

    # Test combined loss
    combined = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    total, ce, dice = combined(logits, targets)
    print(f"\nCombinedLoss:")
    print(f"  CE   = {ce.item():.4f}")
    print(f"  Dice = {dice.item():.4f}")
    print(f"  Total= {total.item():.4f}  (should ≈ CE + Dice = {ce.item()+dice.item():.4f})")

    # Test class weights
    pixel_counts = {0: 170000000, 1: 180000000, 2: 120000,
                    3: 120000, 4: 4000000, 5: 3000000,
                    6: 160000000, 7: 3400000, 8: 2900000,
                    9: 430000, 10: 1600000}
    weights = compute_class_weights(pixel_counts, cfg.NUM_CLASSES, method="median_freq")
    print(f"\nClass weights (median_freq):")
    for i, w in enumerate(weights):
        name = cfg.ACTIVE_CLASSES.get(i, f"class_{i}")
        print(f"  {i:2d} {name:<15s} weight={w:.3f}")

    print("\n  losses.py smoke test PASSED")
