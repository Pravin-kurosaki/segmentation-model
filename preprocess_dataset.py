"""
preprocess_dataset.py — Phase 6: Preprocessing Pipeline

Converts raw CelebAMask-HQ data into model-ready tensors:

  For each image:
    1024x1024 RGB JPG
      -> Resize to 256x256  (bilinear interpolation)
      -> Convert to RGB
      -> Save as PNG
      -> Output: dataset/processed/images/{id:05d}.png

  For each mask:
    Per-part binary PNGs (19 parts per image)
      -> Merge with priority order (coarse first, fine-grained last)
      -> Resize to 256x256  (NEAREST-NEIGHBOR — never bilinear for masks)
      -> Save as single-channel uint8 PNG (values = class IDs 0-10)
      -> Output: dataset/processed/masks/{id:05d}.png

  Splits (by identity when mapping available, else random):
      -> dataset/splits/train.txt  (70%)
      -> dataset/splits/val.txt    (15%)
      -> dataset/splits/test.txt   (15%)

Usage:
    python preprocess_dataset.py                         # full run
    python preprocess_dataset.py --limit 100             # test on 100 images
    python preprocess_dataset.py --workers 4             # parallel processing
    python preprocess_dataset.py --skip-existing         # resume interrupted run
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as cfg

# ── Merge priority (coarse -> fine, so fine-grained classes are never overwritten)
MERGE_PRIORITY = {
    "cloth": 0,  "ear_r": 0,  "neck_l": 0,
    "neck":  1,  "skin":  1,  "face":   1,
    "hair":  2,
    "hat":   3,
    "l_ear": 4,  "r_ear": 4,
    "eye_g": 5,  "eyeglass": 5, "glasses": 5,
    "l_brow": 6, "r_brow": 6,
    "l_eye": 7,  "r_eye": 7,
    "nose":  8,
    "mouth": 9,  "u_lip": 9,  "l_lip": 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# Mask index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mask_index(mask_dir: Path) -> dict:
    """img_id (int) -> list of (part_name, mask_path)"""
    index = defaultdict(list)
    for sub in sorted(mask_dir.iterdir()):
        if not sub.is_dir():
            continue
        for mask_file in sub.glob("*.png"):
            stem = mask_file.stem
            parts = stem.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                img_id = int(parts[0])
                index[img_id].append((parts[1], mask_file))
    return dict(index)


# ─────────────────────────────────────────────────────────────────────────────
# Single image processor (runs in worker process)
# ─────────────────────────────────────────────────────────────────────────────

def process_one(args: tuple) -> dict:
    """
    Process a single image + its masks.
    Returns dict with status info.
    Called by both sequential and parallel paths.
    """
    (img_id, img_path, part_masks,
     out_img_dir, out_mask_dir,
     target_h, target_w, skip_existing) = args

    out_img  = out_img_dir  / f"{img_id:05d}.png"
    out_mask = out_mask_dir / f"{img_id:05d}.png"

    result = {"id": img_id, "ok": False, "skipped": False, "error": None}

    if skip_existing and out_img.exists() and out_mask.exists():
        result["ok"] = True
        result["skipped"] = True
        return result

    try:
        # ── Image ──────────────────────────────────────────────────────────
        raw_img = Image.open(img_path).convert("RGB")
        img_256 = raw_img.resize((target_w, target_h), Image.BILINEAR)
        img_256.save(out_img)

        # ── Mask ───────────────────────────────────────────────────────────
        orig_w, orig_h = raw_img.size

        # Sort by priority (coarse first)
        sorted_parts = sorted(
            part_masks,
            key=lambda x: MERGE_PRIORITY.get(x[0], 5)
        )

        merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for part_name, mask_path in sorted_parts:
            label_id = cfg.CELEBAMASK_LABEL_MAP.get(part_name, None)
            if label_id is None:
                continue
            try:
                pm = np.array(Image.open(mask_path).convert("L"))
                if pm.shape != (orig_h, orig_w):
                    pm = np.array(
                        Image.open(mask_path).convert("L").resize(
                            (orig_w, orig_h), Image.NEAREST
                        )
                    )
                merged[pm > 0] = label_id
            except Exception:
                pass  # skip corrupt part mask

        # Resize mask with NEAREST NEIGHBOR — never bilinear
        mask_pil = Image.fromarray(merged, mode="L")
        mask_256 = mask_pil.resize((target_w, target_h), Image.NEAREST)
        mask_256.save(out_mask)

        result["ok"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Identity-based split
# ─────────────────────────────────────────────────────────────────────────────

def make_identity_splits(
    all_ids: list,
    mapping_file: Path,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split image IDs by CelebA identity to prevent the same person
    appearing in both train and test sets.

    Returns: (train_ids, val_ids, test_ids)
    """
    id_set = set(all_ids)

    # Build identity -> [img_id, ...] map
    identity_to_imgs = defaultdict(list)
    fallback_ids = []

    if mapping_file.exists():
        print(f"  Using identity mapping: {mapping_file.name}")
        with open(mapping_file, "r") as f:
            lines = f.readlines()

        for line in lines[1:]:   # skip header
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    hq_idx   = int(parts[0])
                    celeba_id = int(parts[1])  # orig_idx used as identity
                    if hq_idx in id_set:
                        identity_to_imgs[celeba_id].append(hq_idx)
                except ValueError:
                    pass

        identities = list(identity_to_imgs.keys())
        rng = random.Random(seed)
        rng.shuffle(identities)

        n = len(identities)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_ids = []
        val_ids   = []
        test_ids  = []

        for i, ident in enumerate(identities):
            imgs = identity_to_imgs[ident]
            if i < n_train:
                train_ids.extend(imgs)
            elif i < n_train + n_val:
                val_ids.extend(imgs)
            else:
                test_ids.extend(imgs)

        # Any IDs not in mapping -> add to train
        mapped_ids = set(train_ids + val_ids + test_ids)
        unmapped   = [i for i in all_ids if i not in mapped_ids]
        train_ids.extend(unmapped)

        print(f"  Identities split: {n_train} train / {n_val} val / {n-n_train-n_val} test")

    else:
        print("  No mapping file found — using random split (not identity-based)")
        ids = list(all_ids)
        rng = random.Random(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_ids = ids[:n_train]
        val_ids   = ids[n_train:n_train + n_val]
        test_ids  = ids[n_train + n_val:]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def write_split_file(path: Path, ids: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img_id in sorted(ids):
            f.write(f"{img_id:05d}\n")
    print(f"  Saved: {path.name}  ({len(ids):,} samples)")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    raw_dir:      Path,
    out_img_dir:  Path,
    out_mask_dir: Path,
    splits_dir:   Path,
    target_size:  tuple = (256, 256),
    limit:        int   = 0,
    workers:      int   = 0,
    skip_existing: bool = False,
    seed:         int   = 42,
) -> dict:

    target_h, target_w = target_size

    print("=" * 65)
    print("  CelebAMask-HQ Preprocessing — Phase 6")
    print("=" * 65)

    # ── Locate raw directories ────────────────────────────────────────────────
    hq_root  = raw_dir / "CelebAMask-HQ"
    img_dir  = hq_root / "CelebA-HQ-img"
    mask_dir = hq_root / "CelebAMask-HQ-mask-anno"
    mapping  = hq_root / "CelebA-HQ-to-CelebA-mapping.txt"

    if not img_dir.exists() or not mask_dir.exists():
        print(f"  ERROR: Could not find image/mask directories under {raw_dir}")
        return {}

    print(f"\n  Source images : {img_dir}")
    print(f"  Source masks  : {mask_dir}")
    print(f"  Output images : {out_img_dir}")
    print(f"  Output masks  : {out_mask_dir}")
    print(f"  Target size   : {target_h} x {target_w}")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images ────────────────────────────────────────────────────────
    image_paths = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    if limit:
        image_paths = image_paths[:limit]
        print(f"\n  [DEV] Limiting to first {limit} images")

    n_images = len(image_paths)
    print(f"\n  Total images to process: {n_images:,}")

    # ── Build mask index ──────────────────────────────────────────────────────
    print("  Building mask index...")
    mask_index = build_mask_index(mask_dir)
    print(f"  Mask index built: {len(mask_index):,} image IDs")

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = []
    for img_path in image_paths:
        img_id    = int(img_path.stem) if img_path.stem.isdigit() else None
        if img_id is None:
            continue
        part_masks = mask_index.get(img_id, [])
        tasks.append((
            img_id, img_path, part_masks,
            out_img_dir, out_mask_dir,
            target_h, target_w, skip_existing,
        ))

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"\n  Processing with {'multiprocessing' if workers > 0 else 'single thread'}...")
    t0 = time.time()

    n_ok = 0
    n_skipped = 0
    n_failed  = 0
    errors = []

    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, t): t[0] for t in tasks}
            with tqdm(total=len(tasks), unit="img", desc="  Preprocessing") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    if res["ok"]:
                        if res["skipped"]:
                            n_skipped += 1
                        else:
                            n_ok += 1
                    else:
                        n_failed += 1
                        errors.append(res)
                    pbar.update(1)
                    pbar.set_postfix(ok=n_ok, skip=n_skipped, fail=n_failed)
    else:
        # Single-threaded with tqdm
        for task in tqdm(tasks, unit="img", desc="  Preprocessing"):
            res = process_one(task)
            if res["ok"]:
                if res["skipped"]:
                    n_skipped += 1
                else:
                    n_ok += 1
            else:
                n_failed += 1
                errors.append(res)

    elapsed = time.time() - t0
    total_processed = n_ok + n_skipped

    print(f"\n  Processed  : {n_ok:,}")
    print(f"  Skipped    : {n_skipped:,} (already existed)")
    print(f"  Failed     : {n_failed:,}")
    print(f"  Time       : {elapsed:.1f}s  ({elapsed/max(total_processed,1):.3f}s/image)")

    if errors[:5]:
        print("\n  Sample errors:")
        for e in errors[:5]:
            print(f"    ID {e['id']}: {e['error']}")

    # ── Verify output ─────────────────────────────────────────────────────────
    print("\n  Verifying output...")
    processed_imgs  = list(out_img_dir.glob("*.png"))
    processed_masks = list(out_mask_dir.glob("*.png"))
    print(f"  Images saved  : {len(processed_imgs):,}")
    print(f"  Masks saved   : {len(processed_masks):,}")

    # Quick sanity: open one image and one mask
    if processed_imgs:
        sample_img  = Image.open(processed_imgs[0])
        sample_mask = Image.open(
            out_mask_dir / processed_imgs[0].name
        )
        unique_vals = sorted(set(np.array(sample_mask).flatten().tolist()))
        print(f"  Sample check  : image={sample_img.size} mode={sample_img.mode}")
        print(f"  Sample mask   : size={sample_mask.size} classes={unique_vals}")

    # ── Train/val/test splits ─────────────────────────────────────────────────
    print("\n  Creating train/val/test splits...")
    all_processed_ids = [
        int(p.stem) for p in processed_imgs if p.stem.isdigit()
    ]

    train_ids, val_ids, test_ids = make_identity_splits(
        all_ids      = all_processed_ids,
        mapping_file = mapping,
        train_ratio  = cfg.TRAIN_RATIO,
        val_ratio    = cfg.VAL_RATIO,
        seed         = cfg.RANDOM_SEED,
    )

    write_split_file(splits_dir / "train.txt", train_ids)
    write_split_file(splits_dir / "val.txt",   val_ids)
    write_split_file(splits_dir / "test.txt",  test_ids)

    # ── Save preprocessing config ─────────────────────────────────────────────
    stats = {
        "n_processed":  total_processed,
        "n_failed":     n_failed,
        "n_train":      len(train_ids),
        "n_val":        len(val_ids),
        "n_test":       len(test_ids),
        "target_size":  list(target_size),
        "image_interp": "bilinear",
        "mask_interp":  "nearest",
        "split_by":     "identity" if mapping.exists() else "random",
        "elapsed_s":    round(elapsed, 1),
    }
    stats_path = cfg.METRICS_DIR / "preprocessing_stats.json"
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Processed images : {total_processed:,}")
    print(f"  Train / Val / Test: {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}")
    print(f"  Split method     : {'Identity-based' if mapping.exists() else 'Random'}")
    print(f"  Output size      : {target_h} x {target_w}")
    print(f"  Mask interpolation: NEAREST (correct for segmentation)")
    print(f"  Total time       : {elapsed:.1f}s")
    print(f"  Stats saved      : {stats_path}")
    print("=" * 65)
    print("\n  Next step: python src/dataset.py  (verify DataLoader)")
    print("             python src/train.py    (start training)")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CelebAMask-HQ Preprocessing Pipeline")
    parser.add_argument("--raw-dir",      type=Path, default=cfg.CELEBAMASK_RAW_DIR)
    parser.add_argument("--out-img-dir",  type=Path, default=cfg.PROCESSED_IMAGES_DIR)
    parser.add_argument("--out-mask-dir", type=Path, default=cfg.PROCESSED_MASKS_DIR)
    parser.add_argument("--splits-dir",   type=Path, default=cfg.SPLITS_DIR)
    parser.add_argument("--size",         type=int,  default=256)
    parser.add_argument("--limit",        type=int,  default=0,
                        help="Process only first N images (0=all)")
    parser.add_argument("--workers",      type=int,  default=0,
                        help="Parallel workers (0=single thread, safer on Windows)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that are already processed (resume)")
    args = parser.parse_args()

    preprocess(
        raw_dir       = args.raw_dir,
        out_img_dir   = args.out_img_dir,
        out_mask_dir  = args.out_mask_dir,
        splits_dir    = args.splits_dir,
        target_size   = (args.size, args.size),
        limit         = args.limit,
        workers       = args.workers,
        skip_existing = args.skip_existing,
        seed          = cfg.RANDOM_SEED,
    )
