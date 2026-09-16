"""
config.py — Central configuration for the Occlusion-Aware Face Recognition project.

All paths, hyper-parameters and class definitions live here.
Never hard-code these values anywhere else; import from this module instead.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. Project root — always resolved relative to this file's location
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../face/

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR        = PROJECT_ROOT / "dataset"
RAW_DIR            = DATASET_DIR  / "raw"
PROCESSED_DIR      = DATASET_DIR  / "processed"
SPLITS_DIR         = DATASET_DIR  / "splits"

# Raw dataset sub-directories (one per source)
CELEBAMASK_RAW_DIR = RAW_DIR / "celebamask_hq"
MFSD_RAW_DIR       = RAW_DIR / "mfsd"
OTHER_RAW_DIR      = RAW_DIR / "other"

# Processed output directories
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR  = PROCESSED_DIR / "masks"

# Train / val / test split files (plain-text, one filename stem per line)
TRAIN_SPLIT_FILE = SPLITS_DIR / "train.txt"
VAL_SPLIT_FILE   = SPLITS_DIR / "val.txt"
TEST_SPLIT_FILE  = SPLITS_DIR / "test.txt"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Output directories
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = PROJECT_ROOT / "checkpoints"
RESULTS_DIR       = PROJECT_ROOT / "results"
PREDICTIONS_DIR   = RESULTS_DIR  / "predictions"
VIZ_DIR           = RESULTS_DIR  / "visualizations"
METRICS_DIR       = RESULTS_DIR  / "metrics"

BEST_MODEL_PATH   = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH   = CHECKPOINTS_DIR / "last_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified label taxonomy
#    This is the *target* label set for the whole project.
#    Phase 1 will only use a subset (see ACTIVE_CLASSES below).
# ─────────────────────────────────────────────────────────────────────────────
ALL_CLASSES = {
    0:  "background",
    1:  "skin",
    2:  "left_eye",
    3:  "right_eye",
    4:  "nose",
    5:  "mouth",
    6:  "hair",
    7:  "left_ear",
    8:  "right_ear",
    9:  "glasses",
    10: "sunglasses",
    11: "mask",
    12: "hat",
}

# Classes actually available in CelebAMask-HQ and used for Phase 1 training.
# Adjust this list once additional datasets (mask/sunglasses) are integrated.
ACTIVE_CLASSES = {
    0:  "background",
    1:  "skin",
    2:  "left_eye",
    3:  "right_eye",
    4:  "nose",
    5:  "mouth",
    6:  "hair",
    7:  "left_ear",
    8:  "right_ear",
    9:  "glasses",
    10: "sunglasses",
    11: "mask",
    12: "hat",
}

NUM_CLASSES = len(ACTIVE_CLASSES)   # updated automatically

# Mapping from CelebAMask-HQ original folder/file names -> unified class IDs.
# Keys are the label name fragments used in the dataset annotation files.
CELEBAMASK_LABEL_MAP: dict = {
    # ---- skin / face structure ----
    "skin":           1,
    "neck":           1,   # merge neck into skin for Phase 1
    "face":           1,   # alternative name used in some subsets
    # ---- eyes ----
    "l_eye":          2,
    "left_eye":       2,
    "r_eye":          3,
    "right_eye":      3,
    # ---- brows (merged into eye region for now) ----
    "l_brow":         2,
    "left_brow":      2,
    "r_brow":         3,
    "right_brow":     3,
    # ---- nose ----
    "nose":           4,
    # ---- mouth / lips ----
    "mouth":          5,
    "u_lip":          5,
    "l_lip":          5,
    "upper_lip":      5,
    "lower_lip":      5,
    # ---- hair ----
    "hair":           6,
    # ---- ears ----
    "l_ear":          7,
    "left_ear":       7,
    "r_ear":          8,
    "right_ear":      8,
    # ---- accessories ----
    "eye_g":          9,   # eyeglasses
    "eyeglass":       9,
    "glasses":        9,
    "hat":           12,
    "cloth":          0,   # clothing -> background
    "ear_r":          0,   # ear rings -> background for now
    "neck_l":         0,   # necklace -> background for now
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Image pre-processing
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = (256, 256)          # (H, W) used everywhere
IMAGE_MEAN = (0.485, 0.456, 0.406)   # ImageNet mean (for pretrained encoders)
IMAGE_STD  = (0.229, 0.224, 0.225)   # ImageNet std

# ─────────────────────────────────────────────────────────────────────────────
# 6. Data split ratios
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# 7. Training hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_NAME       = "resnet18"       # "resnet34" is another good choice
ENCODER_WEIGHTS    = "imagenet"       # use pretrained ImageNet weights
ARCHITECTURE       = "unet"

BATCH_SIZE         = 8
NUM_EPOCHS         = 50
LEARNING_RATE      = 1e-4
WEIGHT_DECAY       = 1e-4

# Loss combination weights
CE_LOSS_WEIGHT     = 1.0
DICE_LOSS_WEIGHT   = 1.0

# Checkpointing -- save best model based on this metric
MONITOR_METRIC     = "val_miou"      # options: val_miou | val_dice | val_loss

# ─────────────────────────────────────────────────────────────────────────────
# 8. DataLoader settings
# ─────────────────────────────────────────────────────────────────────────────
NUM_WORKERS        = 0        # set to 0 on Windows to avoid multiprocessing issues
PIN_MEMORY         = True     # faster GPU transfer; set False if CPU-only

# ─────────────────────────────────────────────────────────────────────────────
# 9. Data augmentation (Albumentations) -- probability per transform
# ─────────────────────────────────────────────────────────────────────────────
AUG_HFLIP_P        = 0.5
AUG_ROTATE_LIMIT   = 15       # degrees
AUG_ROTATE_P       = 0.4
AUG_SCALE_LIMIT    = 0.1      # +/-10 % scale
AUG_SHIFT_LIMIT    = 0.05     # +/-5 % shift
AUG_SHIFT_SCALE_P  = 0.4
AUG_BRIGHTNESS     = 0.2
AUG_CONTRAST       = 0.2
AUG_BRIGHT_CONT_P  = 0.4
AUG_BLUR_LIMIT     = 3        # kernel size
AUG_BLUR_P         = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-class colour map for visualisation (RGB tuples, 0-255)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS: dict = {
    0:  (0,   0,   0),    # background  -- black
    1:  (255, 200, 150),  # skin        -- peach
    2:  (0,   128, 255),  # left_eye    -- blue
    3:  (0,   200, 100),  # right_eye   -- green
    4:  (255, 128, 0),    # nose        -- orange
    5:  (220, 50,  50),   # mouth       -- red
    6:  (100, 60,  20),   # hair        -- brown
    7:  (180, 0,   255),  # left_ear    -- purple
    8:  (255, 0,   200),  # right_ear   -- magenta
    9:  (0,   220, 220),  # glasses     -- cyan
    10: (40,  50,  65),   # sunglasses  -- dark slate/charcoal
    11: (0,   190, 210),  # mask        -- surgical teal/cyan
    12: (255, 240, 0),    # hat         -- yellow
}

# ─────────────────────────────────────────────────────────────────────────────
# 11. Occlusion thresholds (FUTURE -- do not use yet)
# ─────────────────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# ─────────────────────────────────────────────────────────────────────────────
# 12. Convenience: ensure all output directories exist when this module loads
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    for d in (
        PROCESSED_IMAGES_DIR,
        PROCESSED_MASKS_DIR,
        SPLITS_DIR,
        CHECKPOINTS_DIR,
        PREDICTIONS_DIR,
        VIZ_DIR,
        METRICS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Quick sanity check -- run as a script to print current configuration
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Occlusion-Aware Face Recognition -- Configuration")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Dataset dir  : {DATASET_DIR}")
    print(f"  Image size   : {IMAGE_SIZE}")
    print(f"  Num classes  : {NUM_CLASSES}")
    print(f"  Encoder      : {ENCODER_NAME} ({ENCODER_WEIGHTS})")
    print(f"  Epochs       : {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    print()
    print("  Active classes:")
    for cid, cname in ACTIVE_CLASSES.items():
        color = CLASS_COLORS.get(cid, (128, 128, 128))
        print(f"    {cid:2d}  {cname:<15s}  RGB{color}")
    print("=" * 60)
