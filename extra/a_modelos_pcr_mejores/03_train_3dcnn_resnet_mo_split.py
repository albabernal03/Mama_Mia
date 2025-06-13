# 03_train_3dcnn_resnet_torchvision.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from pathlib import Path
import random
import json
import os

# TorchVision
import torchvision.models.video as models3d

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path(r"D:\mama_mia_final_corrected")
LABELS_FILE = Path(r"D:\clinical_data_complete.json")
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 5e-4
NUM_WORKERS = 4
SEED = 42
TARGET_SHAPE = (128, 128, 128)

# -----------------------
# SET RANDOM SEED
# -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------
# MODEL - 3D ResNet18
# -----------------------
class ResNet3D18(nn.Module):
    def __init__(self):
        super(ResNet3D18, self).__init__()
        self.backbone = models3d.r3d_18(pretrained=False)
        # Adaptamos input 3 canales
        self.backbone.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # Adaptamos output a binario
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)

# -----------------------
# DATASET
# -----------------------
class MAMAMIA3DDataset(Dataset):
    def __init__(self, data_dir, patient_ids, labels, augment=False, target_shape=(128, 128, 128)):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.labels = labels
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        tensor_path = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
        if not tensor_path.exists():
            tensor_path = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii"

        img = nib.load(tensor_path).get_fdata().astype(np.float32)

        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = self._pad_or_crop_to_shape(img, self.target_shape)

        img = torch.tensor(img).float()
        label = torch.tensor(self.labels[idx]).float()

        return img, label

    def _pad_or_crop_to_shape(self, img, target_shape):
        current_shape = img.shape
        padded_img = img

        for i in range(1, 4):
            current_size = current_shape[i]
            desired_size = target_shape[i - 1]

            if current_size < desired_size:
                diff = desired_size - current_size
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad = [(0, 0)] * 4
                pad[i] = (pad_before, pad_after)
                padded_img = np.pad(padded_img, pad, mode='constant', constant_values=0)

            elif current_size > desired_size:
                start = (current_size - desired_size) // 2
                end = start + desired_size
                if i == 1:
                    padded_img = padded_img[:, start:end, :, :]
                elif i == 2:
                    padded_img = padded_img[:, :, start:end, :]
                elif i == 3:
                    padded_img = padded_img[:, :, :, start:end]

        return padded_img

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print("🔄 Cargando etiquetas...")

    with open(LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)

    pcr_data = {item['patient_id']: item for item in pcr_list}
    patient_ids = []
    labels = []

    print("🔎 Buscando tensores...")

    for pid, info in pcr_data.items():
        pid = pid.strip()
        
        if 'pcr' in info and info['pcr'] in ["0", "1"]:
            tensor_path_niigz = DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
            tensor_path_nii = DATA_DIR / pid / f"{pid}_tensor_3ch.nii"

            if tensor_path_niigz.exists() or tensor_path_nii.exists():
                patient_ids.append(pid)
                labels.append(int(info['pcr']))

    print(f"✅ Datos cargados: {len(patient_ids)} pacientes.")

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        patient_ids, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_dataset = MAMAMIA3DDataset(DATA_DIR, train_ids, train_labels, augment=True, target_shape=TARGET_SHAPE)
    val_dataset = MAMAMIA3DDataset(DATA_DIR, val_ids, val_labels, augment=False, target_shape=TARGET_SHAPE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet3D18()
    model = nn.DataParallel(model)
    model = model.to(device)

    pos_weight = torch.tensor([len(train_labels) / sum(train_labels)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_bal_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        train_preds = []
        train_trues = []
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_trues.extend(targets.detach().cpu().numpy())

        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(imgs)
                val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                val_trues.extend(targets.detach().cpu().numpy())

        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_bal_acc = balanced_accuracy_score(val_trues, val_preds_binary)
        print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f}, Val Balanced Accuracy = {val_bal_acc:.4f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_3dcnn_resnet.pth')
        else:
            patience_counter += 1

        scheduler.step(np.mean(train_losses))

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    print(f"🏆 Best Validation Balanced Accuracy: {best_val_bal_acc:.4f}")
    print("✅ Entrenamiento finalizado.")

