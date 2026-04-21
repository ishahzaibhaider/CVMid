"""Download ~10 curated CIFAKE test images for the live demo.

Uses the Kaggle API. Run once; samples are saved under `app/samples/`.

Usage:
    pip install kaggle          # once
    # put kaggle.json in ~/.kaggle/ (chmod 600)
    python -m scripts.fetch_samples
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "samples"
PER_CLASS = 5


def main() -> None:
    # Don't pre-check creds — locations vary across kaggle CLI versions
    # (~/.kaggle/, ~/.config/kaggle/, env vars). Let the kaggle command itself
    # surface any auth error.
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("Downloading CIFAKE (test set only, ~40 MB)…")
        # -p download location, --unzip extracts automatically.
        # The full dataset is ~300MB but we only need the test/ folder's first few files.
        kaggle_bin = shutil.which("kaggle") or f"{sys.executable} -m kaggle"
        cmd = (
            [kaggle_bin] if " " not in kaggle_bin else kaggle_bin.split()
        ) + [
            "datasets", "download",
            "-d", "birdy654/cifake-real-and-ai-generated-synthetic-images",
            "-p", str(tmp_path),
            "--unzip",
        ]
        subprocess.check_call(cmd)

        # Find test/REAL and test/FAKE anywhere under tmp_path
        real_dir = next(tmp_path.rglob("test/REAL"))
        fake_dir = next(tmp_path.rglob("test/FAKE"))
        print(f"Found:\n  {real_dir}\n  {fake_dir}")

        for label, src in [("real", real_dir), ("fake", fake_dir)]:
            files = sorted(src.glob("*.jpg"))[:PER_CLASS]
            for i, f in enumerate(files, 1):
                dst = OUT / f"{label}_{i:02d}.jpg"
                shutil.copy2(f, dst)
                print(f"  saved {dst.relative_to(ROOT)}")

    print(f"\nDone. {PER_CLASS * 2} samples in {OUT.relative_to(ROOT)}.")
    print("Refresh the Streamlit app — a 'Try a sample' section will appear in the sidebar.")


if __name__ == "__main__":
    main()
