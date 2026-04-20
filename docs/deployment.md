# Deployment Guide — VeriVision

Your 2015 MacBook can't train this in reasonable time. Here's the real plan:

- **Train on Kaggle** (free NVIDIA T4 GPU, CIFAKE already hosted there — zero upload).
- **Deploy the demo on Hugging Face Spaces** (free, Streamlit-native, hosts model weights via Git LFS).

Total cost: **₨ 0**. Total time end-to-end: ~1 hour including waiting for training.

---

## Part 1 — Push the code to GitHub

```bash
cd ~/CVMid
git add -A
git commit -m "initial VeriVision implementation"
# Create a new empty repo on github.com (no README, no .gitignore)
git remote add origin https://github.com/shahzaibhaider/CVMid.git
git branch -M main
git push -u origin main
```

If your GitHub username is different, edit `notebooks/kaggle_train.py` line 22 (`REPO_URL`) to match.

---

## Part 2 — Train on Kaggle (free GPU)

1. Go to <https://www.kaggle.com> → sign in / sign up (free).
2. Top-right **+** → **New Notebook**.
3. Right sidebar panel:
   - **Accelerator** → **GPU T4 x2** (or P100 if offered). Free quota is ~30 hours / week — more than enough.
   - **Internet** → **On** (needed so the notebook can `git clone` and `pip install`).
   - **Add Input** → search `CIFAKE` → pick **"CIFAKE: Real and AI-Generated Synthetic Images"** by `birdy654` → **Add**.
4. Delete the starter cell. Open [`notebooks/kaggle_train.py`](../notebooks/kaggle_train.py) from your repo, copy its contents, paste into a single Kaggle cell.
5. Edit line 22 if your repo URL differs.
6. Click **Run All** (▶). Go make tea — it takes ~10 minutes.
7. When the run finishes:
   - Right sidebar → **Output** → you'll see `verivision_artifacts/` with `custom_cnn.pt`, `resnet50.pt`, training histories, figures, and metrics JSONs.
   - Click the folder → **Download all** → you get a zip.

**Expected output:** `custom_cnn` ≈ 92–93 % accuracy, `resnet50` ≈ 96–97 % accuracy on the CIFAKE test set.

Unzip the downloaded folder into your local project at `models_ckpt/` (the `.pt` files) and `reports/` (the figures + metrics). You now have trained models.

---

## Part 3 — Deploy the Streamlit demo on Hugging Face Spaces

### 3.1 One-time setup

1. Go to <https://huggingface.co> → sign up (free, no card).
2. **Settings** → **Access Tokens** → **New token** → role **Write** → copy it.
3. Install Git LFS locally:
   ```bash
   brew install git-lfs   # macOS
   git lfs install
   ```

### 3.2 Create the Space

1. On HF: click **New** → **Space**.
2. Fill in:
   - **Owner:** your username.
   - **Space name:** `verivision` (or anything).
   - **License:** MIT.
   - **SDK:** **Streamlit**.
   - **Hardware:** CPU basic (free, 2 vCPU / 16 GB).
   - **Public** (so your teacher can see it).
3. Click **Create Space**. HF gives you a git URL like `https://huggingface.co/spaces/YOURUSER/verivision`.

### 3.3 Populate the Space

The simplest way: treat the Space as a second git remote and push a curated subset of files.

```bash
cd ~/CVMid

# Clone the empty Space into a sibling folder
cd ..
git clone https://huggingface.co/spaces/YOURUSER/verivision verivision-space
cd verivision-space

# Copy everything the Space needs
cp -R ../CVMid/src .
cp -R ../CVMid/app .
cp ../CVMid/deploy/app.py .                 # HF entry shim
cp ../CVMid/deploy/requirements.txt .       # HF-tuned deps
cp ../CVMid/deploy/README.md .              # HF Space metadata (frontmatter)
cp ../CVMid/deploy/.gitattributes .         # LFS rules

# Copy the trained checkpoints — LFS will handle these
mkdir -p models_ckpt
cp ../CVMid/models_ckpt/*.pt models_ckpt/

# Tell LFS to track .pt files
git lfs track "*.pt"
git add .gitattributes

# Commit and push
git add -A
git commit -m "deploy VeriVision"
git push
```

When prompted for a password, paste the HF **write token** from step 3.1.

### 3.4 Watch it build

- On your Space page, click the **Logs** tab — you'll see `pip install` then `streamlit run app.py`.
- Cold build takes ~3 minutes (installing torch). Subsequent rebuilds are cached.
- When the log says `You can now view your Streamlit app in your browser` → the **App** tab is live.

### 3.5 Test it

- Open the Space URL.
- Drag any JPG/PNG onto the uploader.
- You should see REAL/FAKE prediction, Grad-CAM heatmap, and FFT spectrum.

---

## Part 4 — Live demo during the presentation

- Open the HF Space URL on the classroom projector — no local setup needed.
- Pre-download 4–5 test images (2 REAL from the CIFAKE test folder, 2 FAKE, and 1 wildcard from the internet) onto a USB or your phone, so you can upload them without depending on classroom wifi speeds.
- Backup plan if HF is slow: run `streamlit run app/streamlit_app.py` locally on the presenter's laptop.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Kaggle notebook says "CIFAKE not found" | You forgot step 2.3 (Add Input). |
| HF Space build times out at `pip install torch` | Ignore — torch wheel is ~800 MB and HF's first build is slow. Refresh the logs in 5 min. |
| HF Space says "No checkpoints" on load | You skipped step 3.3's `cp models_ckpt/*.pt` OR LFS didn't push. Run `git lfs ls-files` to verify. |
| Git push rejected: "blob over 10 MB" | LFS isn't tracking `.pt`. Run `git lfs track "*.pt"` then re-commit. |
| Streamlit says "RuntimeError: mps not available" | HF runs on Linux x86 CPUs; our `resolve_device` auto-falls back to CPU, but if the error persists, re-check that `torch==2.2.0` installed cleanly. |

---

## Cost summary

| Item | Cost |
|---|---|
| Kaggle (GPU training) | Free, 30 h / week quota |
| Hugging Face Spaces (CPU hosting) | Free, unlimited, sleeps after 48 h of no traffic (auto-wakes on first visit) |
| GitHub (source hosting) | Free |
| **Total** | **₨ 0** |
