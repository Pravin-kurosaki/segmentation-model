"""
train.py - Training script for the face segmentation U-Net.

Workflow:
  1. Load config and build DataLoaders
  2. Build U-Net (ResNet18 encoder, pretrained ImageNet)
  3. Initialise Adam optimiser + CombinedLoss
  4. For each epoch:
       a. Train one epoch (tqdm progress bar)
       b. Validate (mIoU, Dice, loss)
       c. Log metrics to CSV + TensorBoard
       d. Save best checkpoint (by val_miou)
       e. Save last checkpoint
  5. Print per-epoch summary

Example output:
  Epoch 001/050  Train Loss: 1.245  Val Loss: 1.102  Val mIoU: 0.61  Val Dice: 0.72
  Epoch 002/050  Train Loss: 1.081  Val Loss: 0.981  Val mIoU: 0.67  Val Dice: 0.76
"""

import sys
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config   as cfg
from model     import build_model, get_device, count_parameters, save_checkpoint
from dataset   import build_dataloaders
from losses    import CombinedLoss
from metrics   import MetricAccumulator


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device : torch.device,
    epoch  : int,
    total_epochs: int,
) -> dict:
    """Run one training epoch. Returns dict of averaged losses."""
    model.train()
    running_loss      = 0.0
    running_ce_loss   = 0.0
    running_dice_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{total_epochs:03d} [Train]",
                unit="batch", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits            = model(images)
        total, ce, dice   = criterion(logits, masks)

        total.backward()
        optimizer.step()

        running_loss      += total.item()
        running_ce_loss   += ce.item()
        running_dice_loss += dice.item()

        pbar.set_postfix(loss=f"{total.item():.4f}")

    pbar.close()

    return {
        "train_loss"     : running_loss      / n_batches,
        "train_ce_loss"  : running_ce_loss   / n_batches,
        "train_dice_loss": running_dice_loss / n_batches,
    }


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device      : torch.device,
    epoch       : int,
    total_epochs: int,
) -> dict:
    """Run one validation pass. Returns dict of losses and metrics."""
    model.eval()
    running_loss      = 0.0
    running_ce_loss   = 0.0
    running_dice_loss = 0.0
    n_batches         = len(loader)

    accumulator = MetricAccumulator(cfg.NUM_CLASSES)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{total_epochs:03d} [Val  ]",
                unit="batch", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)

        logits         = model(images)
        total, ce, dice = criterion(logits, masks)

        running_loss      += total.item()
        running_ce_loss   += ce.item()
        running_dice_loss += dice.item()

        preds = torch.argmax(logits, dim=1)
        accumulator.update(preds.cpu(), masks.cpu())

        pbar.set_postfix(loss=f"{total.item():.4f}")

    pbar.close()

    metrics = accumulator.compute()

    return {
        "val_loss"     : running_loss      / n_batches,
        "val_ce_loss"  : running_ce_loss   / n_batches,
        "val_dice_loss": running_dice_loss / n_batches,
        "val_miou"     : metrics["mean_iou"],
        "val_dice"     : metrics["mean_dice"],
        "val_pixel_acc": metrics["pixel_accuracy"],
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    num_epochs    : int   = cfg.NUM_EPOCHS,
    batch_size    : int   = cfg.BATCH_SIZE,
    learning_rate : float = cfg.LEARNING_RATE,
    weight_decay  : float = cfg.WEIGHT_DECAY,
    encoder_name  : str   = cfg.ENCODER_NAME,
    resume_from   : Path  = None,
    fine_tune_from: Path  = None,
    occlusion_prob: float = 0.0,
    experiment_name: str  = "experiment_001",
) -> None:
    """
    Full training loop.

    Args:
        num_epochs     : Number of training epochs.
        batch_size     : Mini-batch size.
        learning_rate  : Adam learning rate.
        weight_decay   : Adam weight decay.
        encoder_name   : ResNet encoder variant (e.g. 'resnet18').
        resume_from    : Path to checkpoint to resume from (or None).
        experiment_name: Name of the experiment (for results directory).
    """
    device = get_device()
    print("\n" + "=" * 65)
    print("  Occlusion-Aware Face Segmentation — Training")
    print("=" * 65)
    print(f"  Device        : {device}")
    if device.type == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Encoder       : {encoder_name}")
    print(f"  Classes       : {cfg.NUM_CLASSES}")
    print(f"  Image size    : {cfg.IMAGE_SIZE} x {cfg.IMAGE_SIZE}")
    print(f"  Epochs        : {num_epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Learning rate : {learning_rate}")
    print()

    # --- Experiment output directory ---
    exp_dir = cfg.RESULTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # --- DataLoaders ---
    print("  Loading datasets...")
    train_loader, val_loader, _ = build_dataloaders(batch_size=batch_size, occlusion_prob=occlusion_prob)

    # --- Model ---
    print(f"\n  Building U-Net ({encoder_name})...")
    model = build_model(
        encoder_name    = encoder_name,
        encoder_weights = cfg.ENCODER_WEIGHTS,
        num_classes     = cfg.NUM_CLASSES,
    ).to(device)

    params = count_parameters(model)
    print(f"  Total parameters : {params['total']:,}")
    print(f"  Trainable        : {params['trainable']:,}")

    # --- Optimiser ---
    optimizer = optim.Adam(
        model.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay,
    )

    # LR scheduler: reduce on plateau if val mIoU stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # --- Loss ---
    criterion = CombinedLoss(
        ce_weight   = cfg.CE_LOSS_WEIGHT,
        dice_weight = cfg.DICE_LOSS_WEIGHT,
    ).to(device)

    # --- TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = exp_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        use_tb = True
        print(f"  TensorBoard logs : {tb_dir}")
    except Exception as e:
        writer  = None
        use_tb  = False
        print(f"  TensorBoard unavailable: {e}")

    # --- Resume from checkpoint or fine-tune ---
    start_epoch   = 1
    best_val_miou = 0.0

    if resume_from and Path(resume_from).exists():
        print(f"\n  Resuming exact state from: {resume_from}")
        ckpt = torch.load(str(resume_from), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_miou = ckpt.get("val_miou", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best mIoU={best_val_miou:.4f}")
    elif fine_tune_from and Path(fine_tune_from).exists():
        print(f"\n  Fine-tuning weights from: {fine_tune_from}")
        ckpt = torch.load(str(fine_tune_from), map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)

        # Smart transfer for classification head if expanding classes (e.g. 11 -> 13)
        head_w = "segmentation_head.0.weight"
        head_b = "segmentation_head.0.bias"
        if head_w in ckpt["model_state_dict"] and model_dict[head_w].shape != ckpt["model_state_dict"][head_w].shape:
            old_w = ckpt["model_state_dict"][head_w]
            old_b = ckpt["model_state_dict"][head_b]
            # Transfer overlapping classes 0-9
            model_dict[head_w][:10] = old_w[:10]
            model_dict[head_b][:10] = old_b[:10]
            # Transfer hat: old class 10 -> new class 12
            if model_dict[head_w].shape[0] >= 13 and old_w.shape[0] == 11:
                model_dict[head_w][12] = old_w[10]
                model_dict[head_b][12] = old_b[10]
            print("  Smart head transfer: transferred existing classes (0-9 + hat) into expanded 13-class head")

        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} matching layers")
        best_val_miou = 0.0
        start_epoch = 1

    # --- Save config snapshot ---
    config_snapshot = {
        "encoder_name"   : encoder_name,
        "encoder_weights": cfg.ENCODER_WEIGHTS,
        "num_classes"    : cfg.NUM_CLASSES,
        "image_size"     : cfg.IMAGE_SIZE,
        "batch_size"     : batch_size,
        "num_epochs"     : num_epochs,
        "learning_rate"  : learning_rate,
        "weight_decay"   : weight_decay,
        "ce_loss_weight" : cfg.CE_LOSS_WEIGHT,
        "dice_loss_weight": cfg.DICE_LOSS_WEIGHT,
        "active_classes" : {str(k): v for k, v in cfg.ACTIVE_CLASSES.items()},
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # --- Training history CSV ---
    history_path = exp_dir / "training_history.csv"
    history_fieldnames = [
        "epoch", "train_loss", "train_ce_loss", "train_dice_loss",
        "val_loss", "val_ce_loss", "val_dice_loss",
        "val_miou", "val_dice", "val_pixel_acc", "lr", "epoch_time_s",
    ]
    # Open in append mode so resuming adds rows
    history_file = open(history_path, "a", newline="")
    history_writer = csv.DictWriter(history_file, fieldnames=history_fieldnames)
    if start_epoch == 1:
        history_writer.writeheader()

    # --- Epoch loop ---
    print("\n" + "=" * 65)
    print("  Starting training...")
    print("=" * 65)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler step
        scheduler.step(val_metrics["val_miou"])

        # Log
        row = {
            "epoch"          : epoch,
            "train_loss"     : round(train_metrics["train_loss"],      4),
            "train_ce_loss"  : round(train_metrics["train_ce_loss"],   4),
            "train_dice_loss": round(train_metrics["train_dice_loss"],4),
            "val_loss"       : round(val_metrics["val_loss"],          4),
            "val_ce_loss"    : round(val_metrics["val_ce_loss"],       4),
            "val_dice_loss"  : round(val_metrics["val_dice_loss"],     4),
            "val_miou"       : round(val_metrics["val_miou"],          4),
            "val_dice"       : round(val_metrics["val_dice"],          4),
            "val_pixel_acc"  : round(val_metrics["val_pixel_acc"],     4),
            "lr"             : current_lr,
            "epoch_time_s"   : round(epoch_time, 1),
        }
        history_writer.writerow(row)
        history_file.flush()

        if use_tb and writer:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(f"train/{k}" if "train" in k else f"val/{k}", v, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # Per-epoch summary
        print(
            f"  Epoch {epoch:03d}/{num_epochs:03d}  "
            f"Train Loss: {train_metrics['train_loss']:.4f}  "
            f"Val Loss: {val_metrics['val_loss']:.4f}  "
            f"Val mIoU: {val_metrics['val_miou']:.4f}  "
            f"Val Dice: {val_metrics['val_dice']:.4f}  "
            f"[{epoch_time:.0f}s]"
        )

        # Checkpoints
        ckpt_metrics = {
            "val_miou"     : val_metrics["val_miou"],
            "val_dice"     : val_metrics["val_dice"],
            "val_loss"     : val_metrics["val_loss"],
            "val_pixel_acc": val_metrics["val_pixel_acc"],
        }

        # Always save last
        save_checkpoint(
            model, optimizer, epoch, ckpt_metrics,
            cfg.CHECKPOINTS_DIR / "last_model.pth",
        )

        # Save best
        if val_metrics["val_miou"] > best_val_miou:
            best_val_miou = val_metrics["val_miou"]
            save_checkpoint(
                model, optimizer, epoch, ckpt_metrics,
                cfg.CHECKPOINTS_DIR / "best_model.pth",
            )
            print(f"  ** New best model saved (val mIoU = {best_val_miou:.4f}) **")

    history_file.close()
    if use_tb and writer:
        writer.close()

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best val mIoU    : {best_val_miou:.4f}")
    print(f"  Best checkpoint  : {cfg.CHECKPOINTS_DIR / 'best_model.pth'}")
    print(f"  History CSV      : {history_path}")
    print(f"  Config snapshot  : {exp_dir / 'config.json'}")
    print(f"\n  Next: python src/evaluate.py")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face segmentation U-Net")
    parser.add_argument("--epochs",    type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch",     type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--encoder",   type=str,   default=cfg.ENCODER_NAME)
    parser.add_argument("--resume",         type=Path,  default=None)
    parser.add_argument("--fine-tune",      type=Path,  default=None)
    parser.add_argument("--occlusion-prob", type=float, default=0.0)
    parser.add_argument("--exp-name",       type=str,   default="experiment_001")
    args = parser.parse_args()

    train(
        num_epochs     = args.epochs,
        batch_size     = args.batch,
        learning_rate  = args.lr,
        encoder_name   = args.encoder,
        resume_from    = args.resume,
        fine_tune_from = args.fine_tune,
        occlusion_prob = args.occlusion_prob,
        experiment_name= args.exp_name,
    )
