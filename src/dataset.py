import random
"""
dataset.py - PyTorch Dataset and DataLoader factory for face segmentation.

FaceSegDataset loads 256x256 preprocessed images + single-channel class-ID masks.
Albumentations is used for augmentation (image and mask receive identical spatial transforms).
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Albumentations transform pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    """
    Return augmentation pipeline for training.
    All spatial transforms are applied identically to image and mask.
    """
    return A.Compose([
        A.HorizontalFlip(p=cfg.AUG_HFLIP_P),
        A.ShiftScaleRotate(
            shift_limit  = cfg.AUG_SHIFT_LIMIT,
            scale_limit  = cfg.AUG_SCALE_LIMIT,
            rotate_limit = cfg.AUG_ROTATE_LIMIT,
            border_mode  = 0,          # constant padding (black) for masks
            p            = cfg.AUG_SHIFT_SCALE_P,
        ),
        A.RandomBrightnessContrast(
            brightness_limit = cfg.AUG_BRIGHTNESS,
            contrast_limit   = cfg.AUG_CONTRAST,
            p                = cfg.AUG_BRIGHT_CONT_P,
        ),
        A.GaussianBlur(
            blur_limit = (3, cfg.AUG_BLUR_LIMIT),
            p          = cfg.AUG_BLUR_P,
        ),
        # Normalise to ImageNet stats then convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Minimal pipeline for validation / test: just normalise + tensorise.
    No spatial augmentation.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FaceSegDataset(Dataset):
    """
    Loads face images and corresponding single-channel segmentation masks.

    Each mask pixel is an integer class ID in range [0, NUM_CLASSES-1].

    Args:
        split_file  : Path to a text file with one image stem per line (e.g. '00001').
        images_dir  : Directory containing preprocessed RGB images (PNG).
        masks_dir   : Directory containing preprocessed class-ID masks (PNG).
        transform   : Albumentations Compose pipeline (or None for raw arrays).
    """

    def __init__(
        self,
        split_file : Path,
        images_dir : Path,
        masks_dir  : Path,
        transform  : A.Compose = None,
        occlusion_prob : float = 0.0,
    ):
        split_file  = Path(split_file)
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform
        self.occlusion_prob = occlusion_prob

        # Read stem list from split file (strip blank lines)
        with open(split_file) as f:
            self.stems = [line.strip() for line in f if line.strip()]

        # Quick sanity check on first 10 stems (instantaneous startup)
        missing = [
            s for s in self.stems[:10]
            if not (self.images_dir / f"{s}.png").exists()
            or not (self.masks_dir  / f"{s}.png").exists()
        ]
        if missing:
            print(f"  [WARNING] {len(missing)} stems have missing image or mask files "
                  f"(will raise on __getitem__). First few: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        img_path  = self.images_dir / f"{stem}.png"
        mask_path = self.masks_dir  / f"{stem}.png"

        # Load image as uint8 RGB numpy array
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Load mask as uint8 single-channel numpy array (class IDs)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        # Remap hat from 10 to 12 if in 13-class mode (sunglasses=10, mask=11, hat=12)
        if cfg.NUM_CLASSES == 13:
            mask[mask == 10] = 12

        # On-the-fly occlusion augmentation (for Stage B / Phase 2)
        if self.occlusion_prob > 0.0 and random.random() < self.occlusion_prob:
            from occlusion_augmenter import apply_random_occlusion
            image, mask, _ = apply_random_occlusion(image, mask)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # float32 tensor (C, H, W)
            mask  = augmented["mask"]    # long tensor (H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask)

        return {
            "image": image.float(),
            "mask" : mask.long(),
            "stem" : stem,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = None,
    occlusion_prob: float = 0.0,
    config=cfg,
) -> tuple:
    """
    Build train, validation, and test DataLoaders from config paths.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = FaceSegDataset(
        split_file  = config.SPLITS_DIR  / "train.txt",
        images_dir  = config.PROCESSED_IMAGES_DIR,
        masks_dir   = config.PROCESSED_MASKS_DIR,
        transform   = get_train_transforms(),
    )
    val_ds = FaceSegDataset(
        split_file  = config.SPLITS_DIR  / "val.txt",
        images_dir  = config.PROCESSED_IMAGES_DIR,
        masks_dir   = config.PROCESSED_MASKS_DIR,
        transform   = get_val_transforms(),
    )
    test_ds = FaceSegDataset(
        split_file  = config.SPLITS_DIR  / "test.txt",
        images_dir  = config.PROCESSED_IMAGES_DIR,
        masks_dir   = config.PROCESSED_MASKS_DIR,
        transform   = get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size if batch_size is not None else config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size if batch_size is not None else config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size if batch_size is not None else config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )

    print(f"  Train dataset : {len(train_ds):,} images  ({len(train_loader):,} batches)")
    print(f"  Val   dataset : {len(val_ds):,} images  ({len(val_loader):,} batches)")
    print(f"  Test  dataset : {len(test_ds):,} images  ({len(test_loader):,} batches)")

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  dataset.py smoke test")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders()

    batch = next(iter(train_loader))
    images = batch["image"]
    masks  = batch["mask"]

    print(f"\n  Batch image shape : {tuple(images.shape)}")
    print(f"  Batch mask  shape : {tuple(masks.shape)}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Mask  dtype       : {masks.dtype}")
    print(f"  Image value range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mask  classes     : {sorted(masks.unique().tolist())}")

    assert images.shape == (cfg.BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), \
        f"Unexpected image shape: {images.shape}"
    assert masks.shape == (cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), \
        f"Unexpected mask shape: {masks.shape}"
    assert images.dtype == torch.float32
    assert masks.dtype  == torch.int64

    print("\n  All assertions PASSED")
    print("=" * 60)
