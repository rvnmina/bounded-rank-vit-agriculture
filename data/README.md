# Datasets

The datasets are **not** shipped with this repository. This directory is a placeholder; the actual data files are git-ignored (see the top-level `.gitignore`). Place datasets here in the folder layout expected by `src/training.py::TeaPestDataset` and the CIFAR/ImageNet loaders.

## 1. Tea Pest (main experiment)

Expected structure:

```
data/tea_pest/
├── train/
│   ├── Aphids/
│   ├── Mite/
│   ├── Tea_eater_caterpillar/
│   ├── Thrip/
│   └── Mosquito_bug/
├── val/
│   └── (same class subfolders)
└── test/
    └── (same class subfolders)
```

The dataset used in the paper (Mallick et al., 2026, *Sci. Rep.*) has **1,520 field-collected images across 5 tea-pest classes**, augmented to **7,600 samples** during training. Contact the group (Prof. Amlan Chakrabarti / MD Tausif Mallick) for access under the terms agreed with the plantations that provided the images.

## 2. CIFAR-100

`torchvision.datasets.CIFAR100(..., download=True)` will fetch it to `data/cifar100/` automatically the first time `src/cifar100_eval.py` is run.

## 3. ImageNet-1K

ImageNet is **not** downloadable through torchvision. Obtain it directly from https://image-net.org/ and place it as `data/imagenet/{train,val}/`. `src/imagenet_lra.py` uses the `timm` transform stack.

## 4. LRA (Long-Range Arena)

Follow the LRA repository (https://github.com/google-research/long-range-arena) to prepare the five benchmark tasks and place them under `data/lra/`.
