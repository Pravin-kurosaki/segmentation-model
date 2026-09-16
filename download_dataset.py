"""
download_dataset.py
-------------------
Downloads the CelebAMask-HQ dataset from Google Drive using gdown.

Files downloaded:
  1. CelebA-HQ-img.zip        (~15 GB)  — 30,000 face images
  2. CelebAMask-HQ-mask-anno.zip (~2 GB) — pixel-level mask annotations
  3. CelebA-HQ-to-CelebA-mapping.txt    — identity mapping for splits

Output layout expected by src/config.py:
  dataset/raw/celebamask_hq/
      CelebA-HQ-img/          (extracted images: 0.jpg ... 29999.jpg)
      CelebAMask-HQ-mask-anno/ (extracted masks: 0/ ... 14/)
      CelebA-HQ-to-CelebA-mapping.txt
"""

import os
import sys
import zipfile
import gdown
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR    = SCRIPT_DIR / "dataset" / "raw" / "celebamask_hq"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Google Drive file IDs ─────────────────────────────────────────────────────
# These are the individual file IDs inside the CelebAMask-HQ Drive folder.
FILES = [
    {
        "id":       "1X8LfZkjzvqEsFSMezWyiJ4k_NhbCOEFx",
        "filename": "CelebAMask-HQ-mask-anno.zip",
        "desc":     "Mask annotations (~2 GB)",
    },
    {
        "id":       "1badu11NqxGf6qM3PTTooQDJvQbejgbTv",
        "filename": "CelebA-HQ-img.zip",
        "desc":     "Face images (~15 GB)",
        "is_folder": True,   # this ID is a folder on Drive
    },
]

# Individual file IDs sourced from the CelebAMask-HQ repo README
# mask-anno zip
MASK_ANNO_ID  = "1X8LfZkjzvqEsFSMezWyiJ4k_NhbCOEFx"
# images zip (from alternative mirror / Kaggle fallback)
IMG_ZIP_ID    = "1g7DgMiXCTUSggTKFOKmMbgfQnxVNOa0A"   # alt ID often cited
MAPPING_ID    = "1eCdKBR6t6tC4kJM3-gMEzs-pHWQFfkfH"


def download_file(file_id: str, output_path: Path, desc: str) -> bool:
    """Download a single file from Google Drive. Returns True on success."""
    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return True
    print(f"\n  ⬇  Downloading {desc}")
    print(f"     → {output_path}")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        if output_path.exists() and output_path.stat().st_size > 1000:
            print(f"  ✓ Downloaded: {output_path.name}")
            return True
        else:
            print(f"  ✗ Download failed or file too small: {output_path.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip archive. Returns True on success."""
    if not zip_path.exists():
        print(f"  ✗ Zip not found: {zip_path}")
        return False
    print(f"\n  📦 Extracting {zip_path.name} → {extract_to.name}/")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            total = len(z.namelist())
            z.extractall(extract_to)
        print(f"  ✓ Extracted {total} files")
        return True
    except Exception as e:
        print(f"  ✗ Extraction error: {e}")
        return False


def main():
    print("=" * 60)
    print("  CelebAMask-HQ Dataset Downloader")
    print("=" * 60)
    print(f"  Output: {RAW_DIR}\n")

    success_count = 0

    # ── 1. Download mask annotations ──────────────────────────────────────────
    mask_zip = RAW_DIR / "CelebAMask-HQ-mask-anno.zip"
    mask_dir = RAW_DIR / "CelebAMask-HQ-mask-anno"

    if mask_dir.exists() and any(mask_dir.iterdir()):
        print(f"  ✓ Masks already extracted: {mask_dir}")
        success_count += 1
    else:
        ok = download_file(MASK_ANNO_ID, mask_zip, "Mask annotations (~2 GB)")
        if ok:
            mask_dir.mkdir(exist_ok=True)
            if extract_zip(mask_zip, mask_dir):
                success_count += 1
                # Optionally remove zip to save space
                # mask_zip.unlink()

    # ── 2. Download face images ───────────────────────────────────────────────
    img_zip = RAW_DIR / "CelebA-HQ-img.zip"
    img_dir = RAW_DIR / "CelebA-HQ-img"

    if img_dir.exists() and any(img_dir.iterdir()):
        img_count = len(list(img_dir.glob("*.jpg")))
        print(f"  ✓ Images already extracted: {img_count} images in {img_dir}")
        success_count += 1
    else:
        ok = download_file(IMG_ZIP_ID, img_zip, "Face images (~15 GB) — this will take a while")
        if ok:
            img_dir.mkdir(exist_ok=True)
            if extract_zip(img_zip, img_dir):
                success_count += 1

    # ── 3. Download identity mapping ─────────────────────────────────────────
    mapping_file = RAW_DIR / "CelebA-HQ-to-CelebA-mapping.txt"
    if mapping_file.exists():
        print(f"  ✓ Mapping file already exists")
        success_count += 1
    else:
        ok = download_file(MAPPING_ID, mapping_file, "Identity mapping (small)")
        if ok:
            success_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Done: {success_count}/3 components ready")
    if success_count == 3:
        print("  ✓ All dataset components downloaded and extracted!")
        print("  → Run: python inspect_dataset.py")
    else:
        print("  ⚠  Some components failed. Check errors above.")
        print("  → Fallback: download manually from")
        print("    https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv")
    print("=" * 60)


if __name__ == "__main__":
    main()
