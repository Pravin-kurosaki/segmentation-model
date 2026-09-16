"""
download_hf_dataset.py
Downloads CelebAMask-HQ from HuggingFace (no credentials needed).
Source: liusq/CelebAMask-HQ  (~2.94 GB)
"""
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW_DIR = Path("dataset/raw/celebamask_hq")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "CelebAMask-HQ.zip"
ZIP_PATH = RAW_DIR / ZIP_NAME
EXTRACT_DIR = RAW_DIR / "CelebAMask-HQ"

print("=" * 60)
print("  Downloading CelebAMask-HQ from HuggingFace")
print(f"  Size: ~2.94 GB")
print(f"  Output: {RAW_DIR}")
print("=" * 60)

# Step 1: Download
if ZIP_PATH.exists():
    print(f"\n  ✓ Already downloaded: {ZIP_NAME}")
else:
    print(f"\n  >> Downloading {ZIP_NAME} ...")
    downloaded = hf_hub_download(
        repo_id="liusq/CelebAMask-HQ",
        filename=ZIP_NAME,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"  ✓ Saved to: {downloaded}")

# Step 2: Extract
if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
    print(f"\n  ✓ Already extracted: {EXTRACT_DIR}")
else:
    print(f"\n  📦 Extracting {ZIP_NAME} ...")
    EXTRACT_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = z.namelist()
        print(f"     {len(members):,} files to extract...")
        z.extractall(RAW_DIR)
    print(f"  ✓ Extraction complete!")

# Step 3: Report structure
print("\n  📁 Extracted structure:")
for item in sorted(RAW_DIR.iterdir()):
    if item.is_dir():
        count = sum(1 for _ in item.rglob("*") if _.is_file())
        print(f"     {item.name}/  [{count:,} files]")
    elif item.suffix in (".zip", ".txt", ".csv"):
        sz = item.stat().st_size / (1024**2)
        print(f"     {item.name}  ({sz:.1f} MB)")

print("\n  ✓ Done! Run: python inspect_dataset.py")
print("=" * 60)
