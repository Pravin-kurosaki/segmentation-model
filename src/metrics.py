"""
metrics.py - Segmentation evaluation metrics.

All functions accept numpy arrays or torch tensors (converted internally).

Metrics:
  - Pixel Accuracy
  - Per-class IoU  (Intersection over Union)
  - Per-class Dice coefficient
  - Mean IoU
  - Mean Dice
  - compute_all_metrics() -> dict (single call for all metrics)
"""

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy int array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Fraction of correctly classified pixels.

    Args:
        pred   : (N,) or (H, W) or (B, H, W) predicted class IDs
        target : same shape as pred, ground truth class IDs

    Returns:
        Scalar pixel accuracy in [0, 1].
    """
    pred   = _to_numpy(pred).flatten()
    target = _to_numpy(target).flatten()
    return float(np.mean(pred == target))


def iou_per_class(
    pred        : np.ndarray,
    target      : np.ndarray,
    num_classes : int,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Compute per-class IoU (Jaccard index).

    IoU_c = TP_c / (TP_c + FP_c + FN_c)

    Classes with zero TP+FP+FN are assigned NaN (class absent from this batch).

    Args:
        pred, target : flat or multi-dim integer arrays
        num_classes  : total number of classes
        ignore_index : class ID to exclude from computation (-1 to disable)

    Returns:
        np.ndarray of shape (num_classes,) with per-class IoU values (NaN for absent classes).
    """
    pred   = _to_numpy(pred).flatten().astype(np.int64)
    target = _to_numpy(target).flatten().astype(np.int64)

    if ignore_index >= 0:
        valid  = target != ignore_index
        pred   = pred[valid]
        target = target[valid]

    iou = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (target == c)))
        fp = int(np.sum((pred == c) & (target != c)))
        fn = int(np.sum((pred != c) & (target == c)))
        denom = tp + fp + fn
        if denom > 0:
            iou[c] = tp / denom
    return iou


def dice_per_class(
    pred        : np.ndarray,
    target      : np.ndarray,
    num_classes : int,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Compute per-class Dice coefficient.

    Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)

    Args:
        pred, target : flat or multi-dim integer arrays
        num_classes  : total number of classes
        ignore_index : class ID to exclude from computation (-1 to disable)

    Returns:
        np.ndarray of shape (num_classes,) with Dice values (NaN for absent classes).
    """
    pred   = _to_numpy(pred).flatten().astype(np.int64)
    target = _to_numpy(target).flatten().astype(np.int64)

    if ignore_index >= 0:
        valid  = target != ignore_index
        pred   = pred[valid]
        target = target[valid]

    dice = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (target == c)))
        fp = int(np.sum((pred == c) & (target != c)))
        fn = int(np.sum((pred != c) & (target == c)))
        denom = 2 * tp + fp + fn
        if denom > 0:
            dice[c] = (2 * tp) / denom
    return dice


def mean_iou(
    pred        : np.ndarray,
    target      : np.ndarray,
    num_classes : int,
    ignore_index: int = -1,
) -> float:
    """Mean IoU averaged over classes present in target (NaN classes excluded)."""
    per_class = iou_per_class(pred, target, num_classes, ignore_index)
    valid = per_class[~np.isnan(per_class)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def mean_dice(
    pred        : np.ndarray,
    target      : np.ndarray,
    num_classes : int,
    ignore_index: int = -1,
) -> float:
    """Mean Dice averaged over classes present in target (NaN classes excluded)."""
    per_class = dice_per_class(pred, target, num_classes, ignore_index)
    valid = per_class[~np.isnan(per_class)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def compute_all_metrics(
    pred        : np.ndarray,
    target      : np.ndarray,
    num_classes : int,
    class_names : dict = None,
    ignore_index: int  = -1,
) -> dict:
    """
    Compute all metrics in a single call.

    Args:
        pred, target : integer class-ID arrays (any shape, flattened internally)
        num_classes  : total number of classes
        class_names  : {class_id: name} dict for labelling results
        ignore_index : class ID to exclude

    Returns:
        dict with keys:
          pixel_accuracy, mean_iou, mean_dice,
          iou_per_class (np.ndarray), dice_per_class (np.ndarray)
    """
    pxacc   = pixel_accuracy(pred, target)
    iou_cls = iou_per_class(pred, target, num_classes, ignore_index)
    dce_cls = dice_per_class(pred, target, num_classes, ignore_index)

    valid_iou  = iou_cls[~np.isnan(iou_cls)]
    valid_dice = dce_cls[~np.isnan(dce_cls)]

    return {
        "pixel_accuracy"  : pxacc,
        "mean_iou"        : float(np.mean(valid_iou))  if len(valid_iou)  > 0 else 0.0,
        "mean_dice"       : float(np.mean(valid_dice)) if len(valid_dice) > 0 else 0.0,
        "iou_per_class"   : iou_cls,
        "dice_per_class"  : dce_cls,
    }


# ---------------------------------------------------------------------------
# Accumulator for streaming batch updates (used during training/eval loops)
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """
    Streaming confusion matrix metric accumulator.
    Computes exact global IoU, Dice, and Pixel Accuracy across arbitrary-sized datasets
    with O(1) memory overhead.
    """

    def __init__(self, num_classes: int, ignore_index: int = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        """Accumulate confusion matrix counts for a batch."""
        p = _to_numpy(pred).flatten()
        t = _to_numpy(target).flatten()

        mask = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        if self.ignore_index is not None:
            mask &= (t != self.ignore_index)

        p = p[mask]
        t = t[mask]

        hist = np.bincount(
            self.num_classes * t + p, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def compute(self, class_names: dict = None) -> dict:
        """Compute full metrics from accumulated confusion matrix."""
        hist = self.confusion_matrix
        tp = np.diag(hist).astype(np.float64)
        fp = hist.sum(axis=0).astype(np.float64) - tp
        fn = hist.sum(axis=1).astype(np.float64) - tp

        union = tp + fp + fn
        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        valid_u = union > 0
        iou[valid_u] = tp[valid_u] / union[valid_u]

        denom = 2 * tp + fp + fn
        dice = np.full(self.num_classes, np.nan, dtype=np.float64)
        valid_d = denom > 0
        dice[valid_d] = (2 * tp[valid_d]) / denom[valid_d]

        total_pixels = hist.sum()
        pxacc = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0

        valid_iou  = iou[~np.isnan(iou)]
        valid_dice = dice[~np.isnan(dice)]

        return {
            "pixel_accuracy" : pxacc,
            "mean_iou"       : float(np.mean(valid_iou))  if len(valid_iou)  > 0 else 0.0,
            "mean_dice"      : float(np.mean(valid_dice)) if len(valid_dice) > 0 else 0.0,
            "iou_per_class"  : iou,
            "dice_per_class" : dice,
        }

    def reset(self) -> None:
        """Reset accumulated counts."""
        self.confusion_matrix.fill(0)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import config as cfg

    rng = np.random.default_rng(0)
    B, H, W = 4, 256, 256
    C = cfg.NUM_CLASSES

    target = rng.integers(0, C, size=(B, H, W))
    pred   = target.copy()
    noise  = rng.integers(0, C, size=(B, H, W))
    flip   = rng.random((B, H, W)) > 0.85
    pred[flip] = noise[flip]   # introduce ~15% errors

    metrics = compute_all_metrics(pred, target, C, cfg.ACTIVE_CLASSES)

    print(f"Pixel Accuracy : {metrics['pixel_accuracy']:.4f}  (expect ~0.85)")
    print(f"Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"Mean Dice      : {metrics['mean_dice']:.4f}")
    print()
    print(f"  {'Class':>3}  {'Name':<15}  {'IoU':>6}  {'Dice':>6}")
    print(f"  {'-'*40}")
    for c in range(C):
        name = cfg.ACTIVE_CLASSES.get(c, f"class_{c}")
        iou  = metrics['iou_per_class'][c]
        dice = metrics['dice_per_class'][c]
        print(f"  {c:>3}  {name:<15}  {iou:>6.4f}  {dice:>6.4f}")

    # Accumulator test
    acc = MetricAccumulator(C)
    acc.update(pred, target)
    acc.update(pred, target)
    res = acc.compute()
    assert abs(res['mean_iou'] - metrics['mean_iou']) < 1e-6, "Accumulator mismatch"
    print("\n  MetricAccumulator test PASSED")
    print("  metrics.py smoke test PASSED")
