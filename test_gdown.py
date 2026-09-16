"""Test gdown connectivity and find correct file IDs."""
import gdown
from pathlib import Path

raw = Path("dataset/raw/celebamask_hq")
raw.mkdir(parents=True, exist_ok=True)

print("Testing Drive access with known mask-anno file ID...")

# Known IDs referenced across GitHub issues for CelebAMask-HQ
candidate_ids = [
    "1X8LfZkjzvqEsFSMezWyiJ4k_NhbCOEFx",  # mask-anno from various forks
    "1oLetE3HFvIYQ7Ou01KNW8MsR1h3V8gS0",  # alt
]

for fid in candidate_ids:
    print(f"\nTrying ID: {fid}")
    out = raw / "test_dl.tmp"
    try:
        result = gdown.download(
            id=fid,
            output=str(out),
            quiet=False,
            resume=False,
        )
        if result and out.exists():
            sz = out.stat().st_size
            print(f"  SUCCESS — size: {sz} bytes")
            out.unlink()
            break
        else:
            print("  FAILED — returned None or file missing")
            if out.exists():
                out.unlink()
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        if out.exists():
            out.unlink()
