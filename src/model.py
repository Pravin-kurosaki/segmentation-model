"""
model.py — U-Net segmentation model with pretrained ResNet encoder.

Architecture:
  - Library : segmentation-models-pytorch (smp)
  - Model   : U-Net
  - Encoder : ResNet18 pretrained on ImageNet (swappable to ResNet34)
  - Input   : (B, 3, 256, 256)   float32 normalised tensors
  - Output  : (B, NUM_CLASSES, 256, 256)  raw logits  (no softmax/argmax inside)

During inference:
  pred_mask = torch.argmax(logits, dim=1)   # shape: (B, H, W)

Design rules:
  - argmax is NEVER applied inside the model
  - The model returns raw logits so any loss function can be used directly
  - The encoder weights are pretrained; only the decoder trains from scratch
    (encoder also fine-tunes with a lower learning rate — see train.py)
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation-models-pytorch not installed.\n"
        "Run: pip install segmentation-models-pytorch"
    )


def build_model(
    encoder_name:    str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes:     int = 11,
    in_channels:     int = 3,
) -> smp.Unet:
    """
    Build and return a U-Net segmentation model.

    Args:
        encoder_name:    Backbone encoder (e.g. 'resnet18', 'resnet34').
        encoder_weights: Pretrained weights ('imagenet' or None).
        num_classes:     Number of output segmentation classes.
        in_channels:     Number of input image channels (3 for RGB).

    Returns:
        smp.Unet model with raw logit output.
    """
    model = smp.Unet(
        encoder_name    = encoder_name,
        encoder_weights = encoder_weights,
        in_channels     = in_channels,
        classes         = num_classes,
        activation      = None,   # raw logits — never apply activation here
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable, and frozen parameters in the model.

    Returns:
        dict with keys: total, trainable, frozen
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: nn.Module, checkpoint_path, device: torch.device):
    """
    Load model weights from a checkpoint file.

    Args:
        model:           The U-Net model (already built with correct num_classes).
        checkpoint_path: Path to .pth file.
        device:          torch.device to map weights to.

    Returns:
        model with loaded weights, plus the saved metadata dict.
    """
    checkpoint_path = str(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}  "
          f"Val mIoU: {ckpt.get('val_miou', 'N/A'):.4f}" if 'val_miou' in ckpt else "")
    return model, ckpt


def save_checkpoint(
    model:     nn.Module,
    optimizer,
    epoch:     int,
    metrics:   dict,
    path,
) -> None:
    """Save model checkpoint with metadata."""
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **metrics,
    }, str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run as script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import config as cfg

    device = get_device()
    print(f"Device: {device}")

    print(f"\nBuilding U-Net  (encoder={cfg.ENCODER_NAME}, "
          f"classes={cfg.NUM_CLASSES})...")
    model = build_model(
        encoder_name    = cfg.ENCODER_NAME,
        encoder_weights = cfg.ENCODER_WEIGHTS,
        num_classes     = cfg.NUM_CLASSES,
    ).to(device)

    params = count_parameters(model)
    print(f"  Total parameters    : {params['total']:,}")
    print(f"  Trainable           : {params['trainable']:,}")
    print(f"  Frozen              : {params['frozen']:,}")

    # Forward pass smoke test
    dummy = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Input shape  : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(out.shape)}")
    assert out.shape == (2, cfg.NUM_CLASSES, 256, 256), \
        f"Unexpected output shape: {out.shape}"
    print("  Shape assertion: PASSED")

    # Inference demo (argmax outside model)
    pred = torch.argmax(out, dim=1)
    print(f"  Prediction shape : {tuple(pred.shape)}")
    print(f"  Unique class IDs : {sorted(pred.unique().tolist())}")
    print("\n  model.py smoke test PASSED")
