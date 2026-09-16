"""
evaluate.py - Evaluation script for the trained face segmentation model.

Loads the best checkpoint, runs inference on the test set, and reports:
  - Pixel Accuracy
  - Mean IoU
  - Mean Dice
  - Per-class IoU
  - Per-class Dice

Saves results to results/metrics/test_metrics.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config   as cfg
from model     import build_model, get_device, load_checkpoint
from dataset   import build_dataloaders
from metrics   import MetricAccumulator
from visualize import visualize_predictions, plot_training_history


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path : Path,
    split           : str  = "test",
    n_viz           : int  = 16,
    save_viz        : bool = True,
) -> dict:
    """
    Evaluate the model on a dataset split.

    Args:
        checkpoint_path : Path to .pth checkpoint file.
        split           : 'test', 'val', or 'train'.
        n_viz           : Number of sample visualisations to save.
        save_viz        : Whether to save prediction visualisations.

    Returns:
        dict of evaluation metrics.
    """
    device = get_device()
    print(f"\n  Device : {device}")
    print(f"  Checkpoint : {checkpoint_path}")

    # Build model and load weights
    model = build_model(
        encoder_name    = cfg.ENCODER_NAME,
        encoder_weights = None,                # weights loaded from checkpoint
        num_classes     = cfg.NUM_CLASSES,
    ).to(device)
    model, ckpt = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Build dataloaders (we only need the test loader)
    print("\n  Loading dataset...")
    _, val_loader, test_loader = build_dataloaders()
    loader = test_loader if split == "test" else val_loader

    # Accumulate predictions
    print(f"\n  Running inference on {split} set...")
    accumulator = MetricAccumulator(cfg.NUM_CLASSES)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Evaluating [{split}]", unit="batch"):
            images  = batch["image"].to(device, non_blocking=True)
            masks   = batch["mask"]

            logits  = model(images)
            preds   = torch.argmax(logits, dim=1).cpu()

            accumulator.update(preds, masks)

    # Compute metrics
    results = accumulator.compute(class_names=cfg.ACTIVE_CLASSES)

    # Pretty-print results
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {split.upper()} SET")
    print("=" * 60)
    print(f"  Pixel Accuracy : {results['pixel_accuracy']:.4f}  "
          f"({results['pixel_accuracy']*100:.2f}%)")
    print(f"  Mean IoU       : {results['mean_iou']:.4f}")
    print(f"  Mean Dice      : {results['mean_dice']:.4f}")
    print()
    print(f"  {'Class':>3}  {'Name':<15}  {'IoU':>8}  {'Dice':>8}")
    print(f"  {'-'*42}")
    for c in range(cfg.NUM_CLASSES):
        name = cfg.ACTIVE_CLASSES.get(c, f"class_{c}")
        iou  = results["iou_per_class"][c]
        dice = results["dice_per_class"][c]
        iou_str  = f"{iou:.4f}"  if not np.isnan(iou)  else "  N/A  "
        dice_str = f"{dice:.4f}" if not np.isnan(dice) else "  N/A  "
        print(f"  {c:>3}  {name:<15}  {iou_str:>8}  {dice_str:>8}")
    print("=" * 60)

    # Checkpoint info
    if ckpt:
        print(f"\n  Checkpoint epoch       : {ckpt.get('epoch', 'N/A')}")
        if 'val_miou' in ckpt:
            print(f"  Best val mIoU (ckpt)   : {ckpt['val_miou']:.4f}")

    # Serialise for JSON
    metrics_out = {
        "split"          : split,
        "pixel_accuracy" : float(results["pixel_accuracy"]),
        "mean_iou"       : float(results["mean_iou"]),
        "mean_dice"      : float(results["mean_dice"]),
        "per_class"      : {
            cfg.ACTIVE_CLASSES.get(c, f"class_{c}"): {
                "iou"  : float(results["iou_per_class"][c])
                         if not np.isnan(results["iou_per_class"][c]) else None,
                "dice" : float(results["dice_per_class"][c])
                         if not np.isnan(results["dice_per_class"][c]) else None,
            }
            for c in range(cfg.NUM_CLASSES)
        },
        "checkpoint"     : str(checkpoint_path),
    }

    # Save metrics
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.METRICS_DIR / f"{split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved : {metrics_path}")

    # Save visualisations
    if save_viz and n_viz > 0:
        viz_dir = cfg.VIZ_DIR / split
        print(f"\n  Saving {n_viz} prediction visualisations...")
        visualize_predictions(
            model       = model,
            loader      = loader,
            device      = device,
            save_dir    = viz_dir,
            n_samples   = n_viz,
            prefix      = f"eval_{split}",
        )

    # Plot training history if available
    history_csv = cfg.METRICS_DIR.parent / "training_history.csv"
    if history_csv.exists():
        plot_training_history(history_csv, cfg.VIZ_DIR)

    return metrics_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face segmentation model")
    parser.add_argument(
        "--checkpoint",
        type    = Path,
        default = cfg.CHECKPOINTS_DIR / "best_model.pth",
        help    = "Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--split",
        type    = str,
        default = "test",
        choices = ["test", "val", "train"],
        help    = "Dataset split to evaluate on",
    )
    parser.add_argument(
        "--n-viz",
        type    = int,
        default = 16,
        help    = "Number of prediction visualisations to save",
    )
    parser.add_argument(
        "--no-viz",
        action  = "store_true",
        help    = "Skip saving visualisations",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        split           = args.split,
        n_viz           = args.n_viz,
        save_viz        = not args.no_viz,
    )
