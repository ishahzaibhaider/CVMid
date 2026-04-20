"""Paste this into a Kaggle Notebook to train VeriVision on free GPU.

Step-by-step (on kaggle.com):
  1. Create New → Notebook.
  2. Right sidebar → Add Input → search "CIFAKE" → add Jordan Bird's dataset.
  3. Right sidebar → Accelerator → GPU T4 x1 (or P100).
  4. Right sidebar → Internet → ON (so pip install + git clone work).
  5. Paste this whole file into a single cell and Run.
  6. When it finishes, files in /kaggle/working/ can be downloaded (sidebar → Output).

Expected time on a T4:
  - custom_cnn: ~2 min for 10 epochs
  - resnet50:   ~8 min for 5 epochs
"""

# --- 1. Clone the project repo -----------------------------------------------
# Replace with your own fork URL once you push.
REPO_URL = "https://github.com/shahzaibhaider/CVMid.git"  # EDIT ME

import os, subprocess, sys, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)
if not (WORK / "CVMid").exists():
    subprocess.check_call(["git", "clone", REPO_URL])
os.chdir(WORK / "CVMid")
sys.path.insert(0, str(WORK / "CVMid"))

# --- 2. Point the code at Kaggle's mounted CIFAKE dataset --------------------
# Kaggle mounts the dataset (read-only) at /kaggle/input/<slug>/ — find the
# actual train/ and test/ folders wherever they are nested.
def find_cifake_root():
    base = Path("/kaggle/input")
    for train_dir in base.rglob("train"):
        if (train_dir / "REAL").exists() and (train_dir / "FAKE").exists():
            return train_dir.parent
    raise SystemExit("Could not find CIFAKE under /kaggle/input — add the dataset to the notebook.")

src_root = find_cifake_root()
print(f"Found CIFAKE at: {src_root}")

# Symlink into the location the code expects: data/CIFAKE/{train,test}
target = Path("data/CIFAKE")
if target.exists():
    shutil.rmtree(target, ignore_errors=True)
target.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    (target / split).symlink_to(src_root / split)
print("Linked:", list(target.iterdir()))

# --- 3. Install deps (Kaggle already has torch/torchvision/etc) --------------
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "grad-cam"])

# --- 4. Train -----------------------------------------------------------------
# Train BOTH models so you have checkpoints for the demo. Custom CNN first
# because it's fast and a good sanity check.
subprocess.check_call([
    sys.executable, "-m", "scripts.train",
    "--model", "custom_cnn",
    "--epochs", "10",
    "--batch-size", "256",
    "--num-workers", "2",
])

subprocess.check_call([
    sys.executable, "-m", "scripts.train",
    "--model", "resnet50",
    "--epochs", "5",
    "--batch-size", "128",
    "--num-workers", "2",
])

# --- 5. Evaluate both models --------------------------------------------------
for ckpt in ["custom_cnn.pt", "resnet50.pt"]:
    subprocess.check_call([
        sys.executable, "-m", "scripts.evaluate",
        "--checkpoint", f"models_ckpt/{ckpt}",
    ])

# --- 6. Copy artifacts to /kaggle/working for easy download -------------------
OUT = WORK / "verivision_artifacts"
OUT.mkdir(exist_ok=True)
for src in [
    "models_ckpt/custom_cnn.pt",
    "models_ckpt/resnet50.pt",
    "models_ckpt/custom_cnn.history.json",
    "models_ckpt/resnet50.history.json",
]:
    sp = Path(src)
    if sp.exists():
        shutil.copy2(sp, OUT / sp.name)

figures_dst = OUT / "figures"
figures_dst.mkdir(exist_ok=True)
for fig in Path("reports/figures").glob("*.png"):
    shutil.copy2(fig, figures_dst / fig.name)

for metrics in Path("reports").glob("metrics_*.json"):
    shutil.copy2(metrics, OUT / metrics.name)

print("\nDone. Download from Kaggle sidebar → Output → verivision_artifacts/")
print("Files:")
for p in OUT.rglob("*"):
    if p.is_file():
        print(f"  {p.relative_to(WORK)}  ({p.stat().st_size / 1024:.1f} KB)")
