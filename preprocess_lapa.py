import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as cfg

LAPA_DIR = cfg.DATASET_DIR / "LaPa"
TARGET_H, TARGET_W = (cfg.IMAGE_SIZE if isinstance(cfg.IMAGE_SIZE, tuple)
                       else (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

LAPA_TO_PROJECT = np.array([0, 1, 2, 3, 2, 3, 4, 5, 5, 5, 6], dtype=np.uint8)

LAPA_CLASS_NAMES = [
    "background", "skin", "left_eyebrow", "right_eyebrow",
    "left_eye", "right_eye", "nose", "upper_lip", "lower_lip", "inner_mouth", "hair",
]


def process_item(item):
    img_path, lbl_path, out_img, out_mask, skip_existing = item
    out_stem = out_img.stem
    if skip_existing and out_img.exists() and out_mask.exists():
        return {"stem": out_stem, "ok": True, "skipped": True, "error": None}
    try:
        img = Image.open(img_path).convert("RGB")
        img.resize((TARGET_W, TARGET_H), Image.BILINEAR).save(out_img)

        lbl = np.clip(np.array(Image.open(lbl_path).convert("L"), dtype=np.uint8), 0, 10)
        remapped = LAPA_TO_PROJECT[lbl]
        Image.fromarray(remapped, mode="L").resize((TARGET_W, TARGET_H), Image.NEAREST).save(out_mask)
        return {"stem": out_stem, "ok": True, "skipped": False, "error": None}
    except Exception as e:
        return {"stem": out_stem, "ok": False, "skipped": False, "error": str(e)}


def process_split(split_name, limit, skip_existing, out_img_dir, out_mask_dir, num_workers=8):
    split_dir   = LAPA_DIR / split_name
    img_dir_src = split_dir / "images"
    lbl_dir_src = split_dir / "labels"
    if not img_dir_src.exists():
        print(f"  [SKIP] '{split_name}' not found at {img_dir_src}")
        return []

    img_paths = sorted(img_dir_src.glob("*.jpg")) + sorted(img_dir_src.glob("*.png"))
    if limit:
        img_paths = img_paths[:limit]

    print(f"\n  [{split_name.upper()}]  {len(img_paths):,} images (using {num_workers} worker threads)")

    items = []
    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir_src / f"{stem}.png"
        if not lbl_path.exists():
            continue
        out_stem = f"lapa_{split_name}_{stem}"
        items.append((
            img_path,
            lbl_path,
            out_img_dir / f"{out_stem}.png",
            out_mask_dir / f"{out_stem}.png",
            skip_existing
        ))

    good_stems = []
    n_ok = n_skipped = n_failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_item, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(items), desc=f"  {split_name:5s}", unit="img"):
            res = fut.result()
            if res["ok"]:
                good_stems.append(res["stem"])
                if res["skipped"]:
                    n_skipped += 1
                else:
                    n_ok += 1
            else:
                n_failed += 1
                if res["error"]:
                    tqdm.write(f"    ERROR {res['stem']}: {res['error']}")

    print(f"    OK={n_ok}  Skipped={n_skipped}  Failed={n_failed}  ({time.time()-t0:.1f}s, {len(items)/(time.time()-t0):.1f} img/s)")
    return good_stems


def write_split_file(split_path, stems):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        for s in sorted(stems):
            f.write(f"{s}\n")
    print(f"    Wrote {len(stems):,} -> {split_path.name}")


def sanity_check(out_img_dir, out_mask_dir):
    samples = sorted(out_img_dir.glob("lapa_train_*.png"))[:3]
    if not samples:
        print("  [WARN] No lapa_train_* images found.")
        return
    print("\n  Sanity check (first 3):")
    for s in samples:
        mask = np.array(Image.open(out_mask_dir / s.name))
        img  = Image.open(s)
        unique = sorted(set(mask.flatten().tolist()))
        names  = [cfg.ACTIVE_CLASSES.get(c, "?") for c in unique]
        print(f"    {s.name}  img={img.size} mask_classes={list(zip(unique, names))}")


def main(limit=0, skip_existing=False, num_workers=8):
    out_img_dir  = cfg.PROCESSED_IMAGES_DIR
    out_mask_dir = cfg.PROCESSED_MASKS_DIR
    splits_dir   = cfg.SPLITS_DIR
    for d in (out_img_dir, out_mask_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  LaPa Dataset Preprocessing (Multi-threaded)")
    print("=" * 65)
    print(f"  Source       : {LAPA_DIR}")
    print(f"  Target size  : {TARGET_H} x {TARGET_W}")
    print(f"  Limit        : {'ALL' if not limit else limit}")
    print(f"  Threads      : {num_workers}")
    print(f"\n  Class remap (LaPa -> Project):")
    for i, name in enumerate(LAPA_CLASS_NAMES):
        pid = int(LAPA_TO_PROJECT[i])
        print(f"    LaPa {i:2d} ({name:<16s}) -> {pid:2d} ({cfg.ACTIVE_CLASSES.get(pid, '?')})")

    total = 0
    for split_name, split_file in [
        ("train", splits_dir / "train.txt"),
        ("val",   splits_dir / "val.txt"),
        ("test",  splits_dir / "test.txt"),
    ]:
        stems = process_split(split_name, limit, skip_existing, out_img_dir, out_mask_dir, num_workers=num_workers)
        if stems:
            write_split_file(split_file, stems)
            total += len(stems)

    sanity_check(out_img_dir, out_mask_dir)

    print("\n" + "=" * 65)
    print(f"  DONE  Total processed: {total:,} images")
    print("  Next: python src/train.py --exp-name lapa_run_001 --occlusion-prob 0.3")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",         type=int,  default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers",       type=int,  default=8)
    args = parser.parse_args()
    main(limit=args.limit, skip_existing=args.skip_existing, num_workers=args.workers)
