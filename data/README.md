# Data setup — CIFAKE

We use the **CIFAKE** dataset (Bird & Lotfi, IEEE Access 2024):

- 60,000 REAL images (CIFAR-10)
- 60,000 FAKE images (Stable Diffusion v1.4, prompted to match CIFAR-10 classes)
- 32 × 32 RGB
- Splits: 100k train (50k/50k) + 20k test (10k/10k)

## Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# put your kaggle.json in ~/.kaggle/ first (Account → Create New API Token)

cd data
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d CIFAKE
rm cifake-real-and-ai-generated-synthetic-images.zip
```

## Option B — manual download

1. Go to <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
2. Click **Download**
3. Unzip the archive into `data/CIFAKE/`

## Final expected layout

```
data/CIFAKE/
├── train/
│   ├── REAL/       # 50,000 .jpg
│   └── FAKE/       # 50,000 .jpg
└── test/
    ├── REAL/       # 10,000 .jpg
    └── FAKE/       # 10,000 .jpg
```

The code reads from this structure via `torchvision.datasets.ImageFolder`.
Labels are assigned alphabetically: `FAKE → 0`, `REAL → 1`.

## Citation

> Bird, J.J. and Lotfi, A., 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access.
