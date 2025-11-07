# SWIN-CheXpert-Classification

PyTorch implementation of a multi-label chest X‑ray classifier using a Swin Transformer (Swin‑base) backbone trained on the CheXpert dataset. This repository contains training and evaluation scripts (`train.py`, `test.py`) with uncertainty-handling policies for CheXpert labels, AMP training, checkpointing, and multi-checkpoint AUC evaluation.

---

## Contents

* `train.py` — training script (data loading, augmentations, model, loss, checkpointing)
* `test.py` — multi-checkpoint evaluation script and AUC plotting for the 5-class leaderboard

---

## Features / Decisions

* Uses `timm` to load `swin_base_patch4_window7_224` pretrained backbone.
* Multi-label binary classification for 14 CheXpert pathologies.
* Uncertainty handling policies (U-Ones, U-Zeros, U-Ignore, Multiclass stub) configurable per class.
* AMP (automatic mixed precision) training via `torch.cuda.amp` and `GradScaler`.
* Masked BCE loss that ignores `-1` (uncertain) labels when configured.
* Checkpoint resume and periodic saving.
* Multi-checkpoint AUC evaluation script that plots per-class and mean AUC for the CheXpert 5-class leaderboard.

---

## Requirements

* Python 3.8+ recommended
* PyTorch (CUDA-enabled for GPU training)
* torchvision
* timm
* pandas, numpy
* PIL (Pillow)
* scikit-learn
* matplotlib
* tqdm

You can install typical dependencies with:

```bash
pip install torch torchvision timm pandas numpy pillow scikit-learn matplotlib tqdm
```

---

## Repository layout / Expected dataset structure

The scripts assume CSV files with a `Path` column and one column per CheXpert label (names below). Paths in the CSV are relative to the image base directories.

```
UNCOMPRESSED_DIR/
  ├─ train/                 # training images (files referenced in train.csv)
  ├─ valid/                 # validation images (files referenced in valid.csv)
  └─ CheXpert-v1.0 batch 1 (validate & csv)/
       ├─ train.csv
       └─ valid.csv
```

`train.py` default variables (edit at top of file):

* `UNCOMPRESSED_DIR` — base path where `train`/`valid` live
* `TRAIN_IMAGE_BASE`, `VAL_IMAGE_BASE` — directories with images
* `TRAIN_CSV_PATH`, `VAL_CSV_PATH` — CSV files used for dataset
* `CHECKPOINT_DIR` — where checkpoints are saved

`test.py` expects a test CSV and a test image base (edit at top):

* `TEST_IMAGE_BASE`
* `TEST_CSV_PATH`
* `CHECKPOINT_DIR` — directory containing the checkpoints to evaluate

---

## CheXpert labels (order used in scripts)

```
["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
 "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
 "Pleural Other", "Fracture", "Support Devices"]
```

Leaderboard target classes used in `test.py`:

```
["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
```

---

## Uncertainty policy (defaults in scripts)

The repository implements per-class uncertainty handling. By default the following mapping is used (index -> policy):

* `2` (Cardiomegaly): `U-Ones` (treat `-1` as positive)
* `5` (Edema): `U-Ignore` (ignore uncertain labels)
* `6` (Consolidation): `U-Ignore`
* `8` (Atelectasis): `Multiclass` (placeholder — treated as ignore for binary eval)
* `10` (Pleural Effusion): `U-Zeros` (treat `-1` as negative)

You can edit `UNCERTAINTY_POLICIES` in both scripts to change behavior.

---

## How to train

Edit `train.py` top-of-file variables to point to your dataset and checkpoint directory. Then run:

```bash
python train.py
```

Important config knobs in `train.py`:

* `BATCH_SIZE` — default 96 (reduce if out-of-memory)
* `NUM_CLASSES` — must match number of label columns (14 by default)
* `TOTAL_EPOCHS`, `EPOCHS_PER_JOB` — allow splitting long runs into multiple job submissions
* `CHECKPOINT_DIR` — where `checkpoint_epoch_{EPOCH}.pth` files will be saved

The script will attempt to resume from the latest checkpoint automatically if one exists in `CHECKPOINT_DIR`.

---

## How to evaluate / test

1. Prepare a test CSV with the same label columns and a `Path` column pointing to images relative to `TEST_IMAGE_BASE`.
2. Edit `test.py` top variables: `TEST_IMAGE_BASE`, `TEST_CSV_PATH`, `CHECKPOINT_DIR`.
3. Make sure checkpoints in `CHECKPOINT_DIR` follow the naming convention used by the script (e.g. `best_aucfinetune_from_e{EPOCH}.pth` or adapt the filename patterns in the script).
4. Run:

```bash
python test.py
```

The script will load checkpoints, compute per-class ROC AUCs for the five leaderboard classes and save `auc_traces_chexpert.png`.

---

## Loss and masking

The training script uses a masked binary cross-entropy that ignores `-1` labels (or maps them according to the chosen uncertainty policy). This prevents uncertain labels from contributing to loss where policy dictates.

---

## Tips & common edits

* If you get OOM errors: lower `BATCH_SIZE`, enable gradient accumulation, or switch to a smaller model (e.g., `swin_tiny` via `timm`).
* To change augmentations: edit the `transform` in `train.py`. Consider removing heavy ops if CPU becomes a bottleneck.
* To use DistributedDataParallel (DDP) instead of `DataParallel`, replace the `DataParallel` block and follow PyTorch DDP setup instructions.
* To add weighted sampling for class imbalance, create a `WeightedRandomSampler` and pass it into the `DataLoader`.

---

## Checkpoint naming / resume behavior

* `train.py` saves checkpoints as `checkpoint_epoch_{EPOCH}.pth` containing `epoch`, `model_state_dict`, `optimizer_state_dict`, and `loss`.
* On start, the script will look for existing `checkpoint_epoch_*.pth` files and resume from the latest one.
* `test.py` looks for checkpoint files matching `best_aucfinetune_from_e*` by default — update the glob logic if your naming differs.

---
