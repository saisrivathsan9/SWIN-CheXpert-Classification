import os
import torch
import timm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

print("[DEBUG] Starting multi-checkpoint AUC evaluation")

# Paths and config
#CHECKPOINT_DIR = "/scratch/smanika3/checkpointsv3-swin-classification" old
CHECKPOINT_DIR = "/scratch/smanika3/checkpointsv3-swin-classification-stage2_auc_v2" #finetune
MODEL_NAME = "swin_base_patch4_window7_224"
NUM_CLASSES = 14  # Match training checkpoints!
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMAGE_BASE = "/home/smanika3/scratch/CheXpert"
TEST_CSV_PATH = "/home/smanika3/scratch/CheXpert/test_labels.csv"

# All CheXpert CSV/class labels in order
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
# Leaderboard classes
TARGET_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]
target_indices = [CHEXPERT_LABELS.index(cls) for cls in TARGET_COLS]

print(f"[DEBUG] Using device: {DEVICE}")
print(f"[DEBUG] Test image base: {TEST_IMAGE_BASE}")
print(f"[DEBUG] Test CSV: {TEST_CSV_PATH}")
print(f"[DEBUG] Target columns: {TARGET_COLS}, indices: {target_indices}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class CheXpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_paths = df["Path"].tolist()
        self.labels = df["labels"].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Bad file {img_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        if self.transform:
            image = self.transform(image)
        return image, label

UNCERTAINTY_POLICIES = {
    2: "U-Ones",
    5: "U-Ignore",
    6: "U-Ignore",
    8: "Multiclass",   # Multiclass usually acts as U-Ignore for binary eval
    10: "U-Zeros"
    # All others: U-Ignore
}
def policy_label(label, policy):
    if policy == "U-Ignore":
        return label if label != -1 else 0.0
    elif policy == "U-Ones":
        return 1.0 if label == -1 else label
    elif policy == "U-Zeros":
        return 0.0 if label == -1 else label
    else:
        return label # Multiclass—if running binary eval still treat as U-Ignore

def prepare_labels_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] Loaded {len(df)} rows from CSV.")
    # Apply per-class uncertainty logic
    for i, col in enumerate(CHEXPERT_LABELS):
        policy = UNCERTAINTY_POLICIES.get(i, "U-Ignore")
        df[col] = df[col].fillna(0).apply(lambda v: policy_label(v, policy))
    df['Path'] = df['Path'].apply(lambda x: os.path.join(TEST_IMAGE_BASE, x))
    df['labels'] = df[CHEXPERT_LABELS].values.tolist()
    print(f"[DEBUG] Sample image path: {df['Path'].iloc[0]}")
    return df[['Path', 'labels']]

# Prepare test DataLoader
test_df = prepare_labels_dataframe(TEST_CSV_PATH)
test_dataset = CheXpertDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
print(f"[DEBUG] Test DataLoader ready. Batches: {len(test_loader)}")

def extract_epoch(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])
all_ckpts = sorted(
    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.startswith("best_aucfinetune_from_e")],
    key=extract_epoch
) #old = checkpoint_epoch_
print(f"[DEBUG] Found {len(all_ckpts)} checkpoints to evaluate")

auc_traces = {cls: [] for cls in TARGET_COLS}
auc_traces["mean"] = []
epoch_traces = []

# Main evaluation loop
for idx_ckpt, ckpt_path in enumerate(all_ckpts):
    epoch = extract_epoch(ckpt_path)
    print(f"\n======= [EPOCH {epoch}] Evaluating checkpoint {ckpt_path} ({idx_ckpt+1}/{len(all_ckpts)}) =======")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    orig_head = model.head
    model.head = torch.nn.Sequential(orig_head, torch.nn.Dropout(p=0.5))
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc=f"[Epoch {epoch} batches]")):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if batch_idx % 10 == 0:
            print(f"[DEBUG] {batch_idx+1}/{len(test_loader)} batches processed")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(f"[DEBUG] Combined all preds/labels for epoch {epoch}. Shape: {all_preds.shape}")

    # Only leaderboard classes (by column index)
    selected_preds = all_preds[:, target_indices]
    selected_labels = all_labels[:, target_indices]
    aucs = []
    for idx, cls in enumerate(TARGET_COLS):
        try:
            auc = roc_auc_score(selected_labels[:, idx], selected_preds[:, idx])
        except ValueError:
            auc = np.nan
        auc_traces[cls].append(auc)
        aucs.append(auc)
        print(f"[RESULT] {cls}: AUC = {auc:.4f}")
    mean_auc = np.nanmean(aucs)
    auc_traces["mean"].append(mean_auc)
    epoch_traces.append(epoch)
    print(f"[RESULT] Mean 5-class AUC for epoch {epoch}: {mean_auc:.4f}")

# Plotting
print("[DEBUG] Plotting AUC results")
plt.figure(figsize=(10,6))
for cls in TARGET_COLS:
    plt.plot(epoch_traces, auc_traces[cls], marker='o', label=cls)
plt.plot(epoch_traces, auc_traces["mean"], marker='X', linestyle='--', linewidth=2, label="Mean 5-class AUC")
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("CheXpert AUCs per Epoch (5-class leaderboard)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("auc_traces_chexpert.png")
plt.show()
print("[DEBUG] Done!")
