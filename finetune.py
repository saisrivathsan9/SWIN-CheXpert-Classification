# fine-tune.py (fixed: collapse spatial outputs to [B, C] before BSN/loss)
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import datetime

# libAUC loss
from libauc.losses import MultiLabelAUCMLoss

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}, DEVICE: {DEVICE}")

# Which pretraining checkpoints to fine-tune from (your existing CE-trained checkpoints)
TOP_K_EPOCHS = [2, 4, 6]

# Hyperparams / paper-inspired defaults
BATCH_SIZE = 96
NUM_CLASSES = 14
EPOCHS_STAGE2 = 2          # paper used 2 epochs for CheXpert AUC stage; increase if you want
BASE_LR = 0.1              # paper's AUC stage initial LR
MARGIN = 0.8               # paper best margin ~0.8
WEIGHT_DECAY_AUC = 0.0     # paper uses lambda=0 during AUC stage
MOMENTUM = 0.9

# Paths (adjust as needed)
CHECKPOINT_DIR = "/scratch/smanika3/checkpointsv3-swin-classification"   # your original CE checkpoints
FINETUNE_OUTDIR = "/scratch/smanika3/checkpointsv3-swin-classification-stage2_auc_v2"  # new dir for AUC finetunes
os.makedirs(FINETUNE_OUTDIR, exist_ok=True)

UNCOMPRESSED_DIR = "/scratch/smanika3/chexpert/full_uncompressed"
TRAIN_IMAGE_BASE = os.path.join(UNCOMPRESSED_DIR, "train")
CSV_BASE_PATH = os.path.join(UNCOMPRESSED_DIR, 'CheXpert-v1.0 batch 1 (validate & csv)')
TRAIN_CSV_PATH = os.path.join(CSV_BASE_PATH, "train.csv")
VAL_CSV_PATH = os.path.join(CSV_BASE_PATH, "valid.csv")
VAL_IMAGE_BASE = os.path.join(UNCOMPRESSED_DIR, "valid")

# Safety: don't accidentally overwrite your original checkpoint directory
if os.path.abspath(FINETUNE_OUTDIR) == os.path.abspath(CHECKPOINT_DIR):
    raise RuntimeError("FINETUNE_OUTDIR must be different from CHECKPOINT_DIR to avoid overwriting original checkpoints.")

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# ================= UNCERTAINTY POLICY (UNCHANGED FROM YOUR TRAIN.PY) =================
UNCERTAINTY_POLICIES = {
    2: "U-Ones",
    5: "U-Ignore",
    6: "U-Ignore",
    8: "Multiclass",
    10: "U-Zeros"
}
def policy_label(label, policy):
    if policy == "U-Ignore":
        return label  # unchanged
    elif policy == "U-Ones":
        if label == -1:
            return 1.0
        else:
            return label
    elif policy == "U-Zeros":
        if label == -1:
            return 0.0
        else:
            return label
    else:
        return label  # multiclass/advanced policies require more modification

# ================= DATASET (UNCHANGED LABEL HANDLING) =================
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_base, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_base = img_base
        self.transform = transform
        self.is_train = is_train
        for col in CHEXPERT_LABELS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix_to_remove = 'CheXpert-v1.0/train/' if self.is_train else 'CheXpert-v1.0/valid/'
        relative_path = row["Path"].replace(prefix_to_remove, '', 1)
        img_path = os.path.join(self.img_base, relative_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = row[CHEXPERT_LABELS].values.astype(float)
        # Apply uncertainty policy per label (same as your train.py)
        new_labels = []
        for i, v in enumerate(labels):
            policy = UNCERTAINTY_POLICIES.get(i, "U-Ignore")
            new_labels.append(policy_label(v, policy))
        labels = torch.tensor(new_labels, dtype=torch.float32)
        return image, labels

# ================= TRANSFORMS (same as your train.py) =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05), fill=128),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = CheXpertDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_BASE, transform=transform, is_train=True)
val_dataset = CheXpertDataset(VAL_CSV_PATH, VAL_IMAGE_BASE, transform=transform, is_train=False)
print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
print(f"[DEBUG] Validation dataset size: {len(val_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ================= TRAINING UTILITIES =================
scaler = GradScaler()

def validate_model_auc(model, dataloader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="[VALID]"):
            images = images.to(DEVICE)
            logits = model(images)              # raw logits (may still be spatial, handle below)
            # collapse spatial if present
            if logits.dim() > 2:
                # handle common shapes:
                # if [B, C, H, W] -> mean over H,W
                if logits.dim() == 4 and logits.shape[1] == NUM_CLASSES:
                    logits = logits.mean(dim=(2,3))
                # if [B, H, W, C] -> permute then mean
                elif logits.dim() == 4 and logits.shape[-1] == NUM_CLASSES:
                    logits = logits.permute(0,3,1,2).mean(dim=(2,3))
                else:
                    # generic mean over spatial dims (collapse dims 1..-1 except final if final is classes)
                    # fallback: mean over all dims except batch and class if last dim equals NUM_CLASSES
                    if logits.shape[-1] == NUM_CLASSES:
                        spatial_dims = tuple(range(1, logits.dim()-1))
                        logits = logits.mean(dim=spatial_dims)
                    else:
                        # otherwise do global mean over dims 1..
                        logits = logits.mean(dim=tuple(range(1, logits.dim())))
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.vstack(all_logits)   # [N, C]
    labels_policy = np.vstack(all_labels)  # may contain -1
    per_class_auc = []
    n_classes = labels_policy.shape[1]
    for c in range(n_classes):
        mask = (labels_policy[:, c] != -1)
        if mask.sum() == 0:
            per_class_auc.append(np.nan)
            continue
        y_true = labels_policy[mask, c]
        y_true_bin = (y_true > 0).astype(int)
        y_score = logits[mask, c]
        if np.unique(y_true_bin).shape[0] < 2:
            per_class_auc.append(np.nan)
            continue
        try:
            per_class_auc.append(float(roc_auc_score(y_true_bin, y_score)))
        except Exception:
            per_class_auc.append(np.nan)
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else np.nan
    return macro_auc, per_class_auc

# ================= FINE-TUNE LOOP (APPLYING PAPER INGREDIENTS) =================
# Debug print guard - ensure we only print once per run for loss shapes
printed_debug_loss_info = False

for src_epoch in TOP_K_EPOCHS:
    print(f"[FINETUNE] Finetuning checkpoint from epoch {src_epoch}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{src_epoch}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Skipping.")
        continue

    # Create model and load checkpoint (allow partial mismatch)
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # load with strict=False so we can replace head
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Replace head (Linear + Dropout) and re-init weights (paper re-inits classifier head)
    try:
        in_features = model.head.in_features
    except Exception:
        # in some timm versions head is Sequential
        in_features = model.head[0].in_features
    # re-initialize
    new_head = nn.Sequential(
        nn.Linear(in_features, NUM_CLASSES),
        nn.Dropout(0.5),
    )
    # He initialization for new linear
    nn.init.kaiming_normal_(new_head[0].weight, mode='fan_out', nonlinearity='relu')
    if new_head[0].bias is not None:
        nn.init.constant_(new_head[0].bias, 0.0)
    model.head = new_head

    # push to device
    model = model.to(DEVICE)

    # Optimizer: SGD with momentum, weight_decay = 0 (paper's AUC stage)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY_AUC)

    # libAUC loss with paper margin
    auc_loss = MultiLabelAUCMLoss(margin=MARGIN, num_labels=NUM_CLASSES).to(DEVICE)

    best_val_auc = -1.0
    global_step = 0
    total_batches = len(train_loader) * EPOCHS_STAGE2
    print(f"[INFO] Starting AUC fine-tune: epochs={EPOCHS_STAGE2}, approx batches={len(train_loader)} per epoch, total ~{total_batches}")

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[AUC Finetune-epoch{epoch}-from-ckpt{src_epoch}]"):
            images = images.to(DEVICE)
            # convert policy labels to binary targets for training with libAUC
            auc_labels = (labels > 0).float().to(DEVICE)   # -1 and 0 -> 0, positive -> 1

            # adjust LR per global iteration to mimic paper schedule:
            if global_step < 2000:
                lr_factor = 1.0
            elif global_step < 8000:
                lr_factor = 1.0 / 3.0
            else:
                lr_factor = 1.0 / 9.0
            for g in optimizer.param_groups:
                g['lr'] = BASE_LR * lr_factor

            optimizer.zero_grad()
            # Forward pass under autocast to allow mixed precision forward
            with autocast():
                outputs = model(images)                # may be [B, C] or spatial [B, H, W, C] / [B, C, H, W]

                # --- COLLAPSE SPATIAL OUTPUT TO [B, C] if necessary ---
                if outputs.dim() > 2:
                    # Common case: [B, C, H, W] -> mean over H,W
                    if outputs.dim() == 4 and outputs.shape[1] == NUM_CLASSES:
                        outputs = outputs.mean(dim=(2,3))
                    # Alternative: [B, H, W, C] -> permute then mean
                    elif outputs.dim() == 4 and outputs.shape[-1] == NUM_CLASSES:
                        outputs = outputs.permute(0,3,1,2).mean(dim=(2,3))
                    else:
                        # If final dim equals NUM_CLASSES, average spatial dims between batch and final
                        if outputs.shape[-1] == NUM_CLASSES and outputs.dim() >= 3:
                            spatial_dims = tuple(range(1, outputs.dim()-1))
                            outputs = outputs.mean(dim=spatial_dims)
                        else:
                            # fallback: global mean over all non-batch dims
                            outputs = outputs.mean(dim=tuple(range(1, outputs.dim())))

                # --- PAPER: Batch Score Normalization (BSN) ---
                eps = 1e-6
                class_norm = outputs.norm(p=2, dim=0, keepdim=True).clamp(min=eps)
                outputs_bsn = outputs / class_norm

            # --- FIX: cast to float32 and make contiguous OUTSIDE autocast before calling libAUC ---
            outputs_bsn = outputs_bsn.float().contiguous()
            auc_labels = auc_labels.float().contiguous()

            # One-time debug print to show the shapes/dtypes handed to the loss
            if not printed_debug_loss_info:
                print(f"[DEBUG_LOSS] outputs_bsn.shape={outputs_bsn.shape}, outputs_bsn.dtype={outputs_bsn.dtype}, "
                      f"auc_labels.shape={auc_labels.shape}, auc_labels.dtype={auc_labels.dtype}")
                printed_debug_loss_info = True

            # Call the loss in a try/except to print extra diagnostics on failure
            try:
                loss = auc_loss(outputs_bsn, auc_labels)
            except Exception as e:
                # print diagnostics to help debug the internal libAUC failure
                print("=== LIBAUC LOSS ERROR DIAGNOSTICS ===")
                print(f"Exception: {e}")
                print(f"global_step={global_step}, batch_size={outputs_bsn.shape[0]}, num_classes={outputs_bsn.shape[1] if outputs_bsn.dim()>1 else 'NA'}")
                print(f"outputs_bsn.shape={outputs_bsn.shape}, outputs_bsn.stride()={outputs_bsn.stride() if hasattr(outputs_bsn,'stride') else 'NA'}")
                print(f"auc_labels.shape={auc_labels.shape}, auc_labels.stride()={auc_labels.stride() if hasattr(auc_labels,'stride') else 'NA'}")
                print("outputs_bsn sample (clamped):\n", outputs_bsn[:min(4, outputs_bsn.shape[0]), :min(8, outputs_bsn.shape[1])])
                print("auc_labels sample:\n", auc_labels[:min(4, auc_labels.shape[0]), :min(8, auc_labels.shape[1])])
                raise

            scaler.scale(loss).backward()
            # gradient clipping may help stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            global_step += 1

        avg_train_loss = running_loss / len(train_loader.dataset)
        print(f"[AUC Finetune-ckpt{src_epoch}] Epoch {epoch}/{EPOCHS_STAGE2} - Avg Train Loss: {avg_train_loss:.6f}")

        # Validate by AUC (mask -1 as uncertain)
        val_macro_auc, per_class_auc = validate_model_auc(model, val_loader)
        print(f"[AUC Finetune-ckpt{src_epoch}] Epoch {epoch} VAL macro AUC: {val_macro_auc:.6f}")
        for i, a in enumerate(per_class_auc):
            s = f"{a:.6f}" if not np.isnan(a) else "nan"
            print(f"  {CHEXPERT_LABELS[i]:25s}: {s}")

        # Save checkpoint (named with val AUC and source epoch) to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_ckpt = os.path.join(FINETUNE_OUTDIR, f"checkpoint_aucfinetune_from_e{src_epoch}_ep{epoch}_valAUC{val_macro_auc:.6f}_{timestamp}.pth")
        torch.save({
            'epoch': epoch,
            'from_epoch': src_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_macro_auc': val_macro_auc,
            'per_class_auc': per_class_auc
        }, out_ckpt)
        print(f"[AUC Finetune-ckpt{src_epoch}] Saved {out_ckpt}")

        # Keep best by validation macro AUC (separate best file)
        if not np.isnan(val_macro_auc) and val_macro_auc > best_val_auc:
            best_val_auc = val_macro_auc
            best_path = os.path.join(FINETUNE_OUTDIR, f"best_aucfinetune_from_e{src_epoch}.pth")
            torch.save({
                'epoch': epoch,
                'from_epoch': src_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_macro_auc': val_macro_auc,
                'per_class_auc': per_class_auc
            }, best_path)
            print(f"[AUC Finetune-ckpt{src_epoch}] NEW BEST saved to {best_path}")

print("[INFO] AUC fine-tuning finished for all specified source checkpoints.")
