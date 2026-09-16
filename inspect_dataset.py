"""
inspect_dataset.py — Phase 5: Dataset Inspection for CelebAMask-HQ

Performs a thorough audit of the raw dataset BEFORE any training:
  1. Locates all images and masks
  2. Verifies every image has a corresponding merged mask
  3. Checks image dimensions
  4. Reports unique class IDs found in masks
  5. Reports class frequency (pixel count per class)
  6. Detects missing / corrupted files
  7. Saves random image/mask pair visualisations
  8. Prints a clean statistics report

Run this script BEFORE running any preprocessing or training.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --samples 10 --save-viz
"""

import argparse
import sys
import os
import random
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# ── Add project src to path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as cfg

# ── Constants ─────────────────────────────────────────────────────────────────
# CelebAMask-HQ raw annotation labels (19 classes from the original dataset)
# These are the actual folder/file name fragments in CelebAMask-HQ-mask-anno/
CELEBAMASK_PARTS = [
    "skin", "nose", "eye_g", "l_eye", "r_eye",
    "l_brow", "r_brow", "l_ear", "r_ear",
    "mouth", "u_lip", "l_lip",
    "hair", "hat", "ear_r", "neck_l", "neck", "cloth",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_dataset_root(raw_dir: Path) -> dict:
    """
    Auto-detect the image and mask directories inside raw_dir.
    Handles several common extraction layouts from CelebAMask-HQ.
    Returns dict with keys: 'img_dir', 'mask_dir', 'layout'
    """
    # Layout A: flat zip from HuggingFace liusq/CelebAMask-HQ
    #   raw_dir/CelebAMask-HQ/CelebA-HQ-img/  + CelebAMask-HQ-mask-anno/
    hq_root = raw_dir / "CelebAMask-HQ"
    if hq_root.exists():
        img_candidate  = hq_root / "CelebA-HQ-img"
        mask_candidate = hq_root / "CelebAMask-HQ-mask-anno"
        if img_candidate.exists() and mask_candidate.exists():
            return {"img_dir": img_candidate, "mask_dir": mask_candidate, "layout": "A"}

    # Layout B: directly extracted
    img_candidate  = raw_dir / "CelebA-HQ-img"
    mask_candidate = raw_dir / "CelebAMask-HQ-mask-anno"
    if img_candidate.exists() and mask_candidate.exists():
        return {"img_dir": img_candidate, "mask_dir": mask_candidate, "layout": "B"}

    # Layout C: inside CelebAMask-HQ-master (GitHub repo with data)
    master = raw_dir / "CelebAMask-HQ-master"
    img_candidate  = master / "CelebA-HQ-img"
    mask_candidate = master / "CelebAMask-HQ-mask-anno"
    if img_candidate.exists() and mask_candidate.exists():
        return {"img_dir": img_candidate, "mask_dir": mask_candidate, "layout": "C"}

    return {"img_dir": None, "mask_dir": None, "layout": None}


def load_images(img_dir: Path) -> list:
    """Return sorted list of image paths (.jpg / .png)."""
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    return sorted(imgs, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def build_mask_index(mask_dir: Path) -> dict:
    """
    Build index: image_id (int) -> list of (part_name, mask_path).
    CelebAMask-HQ stores masks in sub-folders 0..14, each covering 2000 images.
    Filenames: 00000_skin.png, 00001_l_eye.png, etc.
    """
    index = defaultdict(list)
    for sub in sorted(mask_dir.iterdir()):
        if not sub.is_dir():
            continue
        for mask_file in sorted(sub.glob("*.png")):
            stem = mask_file.stem        # e.g. "00000_skin"
            parts = stem.split("_", 1)  # ["00000", "skin"]
            if len(parts) == 2 and parts[0].isdigit():
                img_id = int(parts[0])
                part   = parts[1]
                index[img_id].append((part, mask_file))
    return dict(index)


# Priority order for mask merging: LOWER number = applied FIRST (can be overwritten).
# Coarse/large regions first, fine-grained specific regions last.
# This prevents skin (large) from overwriting nose/eyes (small).
MERGE_PRIORITY = {
    # background-like → lowest priority (applied first)
    "cloth":    0, "ear_r":   0, "neck_l":  0,
    # skin / neck → applied second
    "neck":     1, "skin":    1, "face":    1,
    # hair
    "hair":     2,
    # hat
    "hat":      3,
    # ears
    "l_ear":    4, "r_ear":   4,
    # eyeglasses
    "eye_g":    5, "eyeglass": 5, "glasses": 5,
    # eyebrows
    "l_brow":   6, "r_brow":  6, "left_brow": 6, "right_brow": 6,
    # eyes
    "l_eye":    7, "r_eye":   7, "left_eye": 7, "right_eye": 7,
    # nose — applied AFTER skin so it is never overwritten
    "nose":     8,
    # mouth / lips — highest priority
    "mouth":    9, "u_lip":   9, "l_lip":   9,
    "upper_lip": 9, "lower_lip": 9,
}


def merge_masks_for_image(img_id: int, part_masks: list, h: int, w: int) -> np.ndarray:
    """
    Merge all part masks for one image into a single H x W class-ID mask.
    Uses cfg.CELEBAMASK_LABEL_MAP for label remapping.

    IMPORTANT: parts are sorted by MERGE_PRIORITY so that coarse regions
    (skin, hair) are written first and fine-grained regions (nose, eyes, mouth)
    are written last — preventing the large skin mask from overwriting nose.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    # Sort by merge priority (ascending) — lower priority applied first
    sorted_parts = sorted(
        part_masks,
        key=lambda x: MERGE_PRIORITY.get(x[0], 5)  # unknown parts → mid-priority
    )

    merged = np.zeros((h, w), dtype=np.uint8)
    for part_name, mask_path in sorted_parts:
        label_id = cfg.CELEBAMASK_LABEL_MAP.get(part_name, None)
        if label_id is None:
            continue  # unknown part — skip
        try:
            part_mask = np.array(PILImage.open(mask_path).convert("L"))
            if part_mask.shape != (h, w):
                part_mask = np.array(
                    PILImage.open(mask_path).convert("L").resize(
                        (w, h), PILImage.NEAREST
                    )
                )
            merged[part_mask > 0] = label_id
        except Exception:
            pass  # skip corrupted mask files
    return merged


def check_image(img_path: Path) -> dict:
    """Open image, return basic info. Returns None on error."""
    try:
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        mode = img.mode
        return {"path": img_path, "w": w, "h": h, "mode": mode, "ok": True}
    except Exception as e:
        return {"path": img_path, "ok": False, "error": str(e)}


# ── Main inspection ───────────────────────────────────────────────────────────

def inspect(raw_dir: Path, n_samples: int = 8, save_viz: bool = True,
            max_check: int = 500) -> dict:
    """
    Full dataset inspection. Returns a statistics dict.
    max_check: max images to open for pixel-level class stats (slow on 30k).
    """
    from PIL import Image

    print("=" * 65)
    print("  CelebAMask-HQ Dataset Inspection — Phase 5")
    print("=" * 65)

    # 1. Locate dataset
    print("\n[1/7] Locating dataset directories...")
    dirs = find_dataset_root(raw_dir)
    img_dir  = dirs["img_dir"]
    mask_dir = dirs["mask_dir"]
    layout   = dirs["layout"]

    if img_dir is None or mask_dir is None:
        print("\n  ERROR: Could not find image or mask directories.")
        print(f"  Searched inside: {raw_dir}")
        print("\n  Expected one of these layouts:")
        print("    raw_dir/CelebAMask-HQ/CelebA-HQ-img/")
        print("    raw_dir/CelebA-HQ-img/")
        print("\n  Please check your extraction and re-run.")
        return {}

    print(f"  Layout detected   : {layout}")
    print(f"  Images directory  : {img_dir}")
    print(f"  Masks directory   : {mask_dir}")

    # 2. Collect images
    print("\n[2/7] Collecting images...")
    image_paths = load_images(img_dir)
    n_images = len(image_paths)
    print(f"  Images found: {n_images:,}")

    if n_images == 0:
        print("  ERROR: No images found. Check img_dir path.")
        return {}

    # 3. Collect masks
    print("\n[3/7] Building mask index...")
    mask_index = build_mask_index(mask_dir)
    n_mask_ids = len(mask_index)
    print(f"  Image IDs with masks: {n_mask_ids:,}")

    # 4. Cross-check images vs masks
    print("\n[4/7] Cross-checking images vs masks...")
    image_ids = set()
    for p in image_paths:
        if p.stem.isdigit():
            image_ids.add(int(p.stem))

    mask_ids = set(mask_index.keys())
    ids_with_masks    = image_ids & mask_ids
    ids_missing_masks = image_ids - mask_ids
    ids_orphan_masks  = mask_ids  - image_ids

    print(f"  Images with masks   : {len(ids_with_masks):,}")
    print(f"  Images missing masks: {len(ids_missing_masks):,}")
    print(f"  Orphan mask IDs     : {len(ids_orphan_masks):,}")

    if ids_missing_masks:
        sample = sorted(ids_missing_masks)[:10]
        print(f"  Sample missing: {sample}")

    # 5. Image dimension check (sample)
    print("\n[5/7] Checking image dimensions (sampling 200 images)...")
    sample_paths = random.sample(image_paths, min(200, n_images))
    dim_counter = Counter()
    mode_counter = Counter()
    corrupted = []

    for p in sample_paths:
        info = check_image(p)
        if info["ok"]:
            dim_counter[(info["h"], info["w"])] += 1
            mode_counter[info["mode"]] += 1
        else:
            corrupted.append(str(p))

    print(f"  Corrupted images (in sample): {len(corrupted)}")
    print(f"  Dimensions found:")
    for (h, w), cnt in dim_counter.most_common(5):
        print(f"    {h} x {w}: {cnt} images")
    print(f"  Color modes: {dict(mode_counter)}")

    # 6. Class frequency analysis (sample up to max_check images)
    print(f"\n[6/7] Analysing class pixel frequencies (up to {max_check} images)...")
    pixel_counts   = defaultdict(int)
    parts_seen     = defaultdict(int)
    check_ids = sorted(ids_with_masks)[:max_check]

    # Get reference image size for merging
    ref_img = Image.open(image_paths[0])
    ref_h, ref_w = ref_img.height, ref_img.width

    for img_id in check_ids:
        part_masks = mask_index[img_id]
        merged = merge_masks_for_image(img_id, part_masks, ref_h, ref_w)
        unique, counts = np.unique(merged, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pixel_counts[int(cls)] += int(cnt)
        for part_name, _ in part_masks:
            parts_seen[part_name] += 1

    total_pixels = sum(pixel_counts.values())

    print(f"  Analysed {len(check_ids)} images")
    print(f"  Unique class IDs found: {sorted(pixel_counts.keys())}")
    print()
    print(f"  {'Class ID':<10} {'Class Name':<18} {'Pixels':>14} {'%':>8}")
    print(f"  {'-'*54}")
    for cls_id in sorted(pixel_counts.keys()):
        cls_name = cfg.ACTIVE_CLASSES.get(cls_id, f"class_{cls_id}")
        px = pixel_counts[cls_id]
        pct = 100.0 * px / total_pixels if total_pixels > 0 else 0
        print(f"  {cls_id:<10} {cls_name:<18} {px:>14,} {pct:>7.2f}%")

    print()
    print(f"  Raw parts seen in masks:")
    for part, cnt in sorted(parts_seen.items(), key=lambda x: -x[1]):
        mapped = cfg.CELEBAMASK_LABEL_MAP.get(part, "UNMAPPED")
        print(f"    {part:<15} -> class {mapped:<4} ({cnt} images)")

    # 7. Save visualisations
    viz_paths = []
    if save_viz and len(ids_with_masks) > 0:
        print(f"\n[7/7] Saving {n_samples} sample visualisations...")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            cfg.VIZ_DIR.mkdir(parents=True, exist_ok=True)
            sample_ids = random.sample(sorted(ids_with_masks), min(n_samples, len(ids_with_masks)))

            for i, img_id in enumerate(sample_ids):
                img_path  = img_dir / f"{img_id}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{img_id}.png"
                if not img_path.exists():
                    continue

                part_masks = mask_index[img_id]
                img_arr    = np.array(Image.open(img_path).convert("RGB"))
                h, w       = img_arr.shape[:2]
                merged     = merge_masks_for_image(img_id, part_masks, h, w)

                # Build colour mask
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for cls_id, rgb in cfg.CLASS_COLORS.items():
                    color_mask[merged == cls_id] = rgb

                # Overlay (alpha blend)
                alpha   = 0.45
                overlay = (img_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Image ID: {img_id}", fontsize=13, fontweight="bold")

                axes[0].imshow(img_arr);    axes[0].set_title("Original Image");    axes[0].axis("off")
                axes[1].imshow(color_mask); axes[1].set_title("Segmentation Mask"); axes[1].axis("off")
                axes[2].imshow(overlay);    axes[2].set_title("Overlay");           axes[2].axis("off")

                # Legend
                patches = [
                    mpatches.Patch(color=[c/255 for c in rgb], label=name)
                    for cls_id, (name, rgb) in {
                        k: (cfg.ACTIVE_CLASSES[k], cfg.CLASS_COLORS[k])
                        for k in sorted(cfg.ACTIVE_CLASSES.keys())
                        if k in cfg.CLASS_COLORS
                    }.items()
                    if np.any(merged == cls_id)
                ]
                if patches:
                    fig.legend(handles=patches, loc="lower center",
                               ncol=min(len(patches), 6), fontsize=8,
                               bbox_to_anchor=(0.5, -0.05))

                save_path = cfg.VIZ_DIR / f"inspect_sample_{i:02d}_id{img_id}.png"
                plt.tight_layout()
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                viz_paths.append(save_path)
                print(f"  Saved: {save_path.name}")

        except Exception as e:
            print(f"  Visualisation error: {e}")
    else:
        print("\n[7/7] Skipping visualisations (--save-viz not set or no matched pairs)")

    # ── Final report ─────────────────────────────────────────────────────────
    stats = {
        "n_images":         n_images,
        "n_mask_ids":       n_mask_ids,
        "ids_with_masks":   len(ids_with_masks),
        "missing_masks":    len(ids_missing_masks),
        "corrupted":        len(corrupted),
        "image_dimensions": {f"{h}x{w}": cnt for (h, w), cnt in dim_counter.most_common()},
        "pixel_counts":     {cfg.ACTIVE_CLASSES.get(k, str(k)): v for k, v in pixel_counts.items()},
        "n_classes_found":  len(pixel_counts),
    }

    print()
    print("=" * 65)
    print("  DATASET STATISTICS SUMMARY")
    print("=" * 65)
    print(f"  Total images          : {n_images:,}")
    print(f"  Images with masks     : {len(ids_with_masks):,}")
    print(f"  Images missing masks  : {len(ids_missing_masks):,}")
    print(f"  Corrupted files       : {len(corrupted)}")
    print(f"  Unique classes found  : {len(pixel_counts)}")
    print(f"  Most common size      : {dim_counter.most_common(1)[0][0] if dim_counter else 'N/A'}")
    print(f"  Visualisations saved  : {len(viz_paths)}")
    if viz_paths:
        print(f"  Viz directory         : {cfg.VIZ_DIR}")
    print("=" * 65)

    # Save stats to JSON
    stats_path = cfg.METRICS_DIR / "dataset_inspection.json"
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")

    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CelebAMask-HQ Dataset Inspector")
    parser.add_argument("--raw-dir",    type=Path,
                        default=cfg.CELEBAMASK_RAW_DIR,
                        help="Path to raw CelebAMask-HQ directory")
    parser.add_argument("--samples",   type=int, default=8,
                        help="Number of random samples to visualise")
    parser.add_argument("--max-check", type=int, default=500,
                        help="Max images to analyse for class frequency")
    parser.add_argument("--save-viz",  action="store_true", default=True,
                        help="Save visualisation images (default: True)")
    parser.add_argument("--no-viz",    action="store_true",
                        help="Disable visualisation saving")
    args = parser.parse_args()

    save_viz = args.save_viz and not args.no_viz

    t0 = time.time()
    stats = inspect(
        raw_dir   = args.raw_dir,
        n_samples = args.samples,
        save_viz  = save_viz,
        max_check = args.max_check,
    )
    elapsed = time.time() - t0
    print(f"\n  Inspection completed in {elapsed:.1f}s")
