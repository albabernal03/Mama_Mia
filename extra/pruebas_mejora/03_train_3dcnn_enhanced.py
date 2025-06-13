# 03_train_3dcnn_enhanced.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path
import random
import json
import os
import math
import timm

# For 3D augmentations
import torchio as tio

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path(r"D:\mama_mia_final_corrected")
LABELS_FILE = Path(r"D:\clinical_data_complete.json")
BATCH_SIZE = 16  # Increased for smaller models
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42
TARGET_SHAPE = (96, 96, 96)  # Smaller for faster training
GRADIENT_CLIP = 1.0
USE_EMA = True
EMA_DECAY = 0.9999

# Model options
MODEL_TYPE = "small"  # "small", "medium", "large"
INPUT_CHANNELS = 3  # 1, 2, or 3 channels

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
# EXPONENTIAL MOVING AVERAGE
# -----------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# -----------------------
# COSINE ANNEALING WITH WARMUP
# -----------------------
class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer)

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# -----------------------
# MODELS
# -----------------------
class SmallResNet3D(nn.Module):
    """Lightweight 3D ResNet for fast experimentation"""
    def __init__(self, input_channels=3, num_classes=1):
        super(SmallResNet3D, self).__init__()
        
        # Much smaller architecture
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Simplified residual blocks
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MediumResNet3D(nn.Module):
    """Medium-sized 3D ResNet"""
    def __init__(self, input_channels=3):
        super(MediumResNet3D, self).__init__()
        # Your existing ResNet3D18 but with configurable input channels
        import torchvision.models.video as models3d
        self.backbone = models3d.r3d_18(pretrained=False)
        
        # Adapt input channels
        self.backbone.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                         stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # Binary output
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)

def get_model(model_type="small", input_channels=3):
    if model_type == "small":
        return SmallResNet3D(input_channels=input_channels)
    elif model_type == "medium":
        return MediumResNet3D(input_channels=input_channels)
    else:  # large - your original
        return MediumResNet3D(input_channels=input_channels)

# -----------------------
# ENHANCED DATASET WITH AUGMENTATIONS
# -----------------------
class MAMAMIA3DDatasetEnhanced(Dataset):
    def __init__(self, data_dir, patient_ids, labels, augment=False, target_shape=(96, 96, 96), input_channels=3):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.labels = labels
        self.target_shape = target_shape
        self.input_channels = input_channels
        
        # Define augmentations using TorchIO
        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), p=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=(-10, 10),
                    translation=(-5, 5),
                    p=0.5
                ),
                tio.RandomNoise(std=(0, 0.1), p=0.3),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
                tio.RandomBiasField(coefficients=0.5, p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        tensor_path = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
        if not tensor_path.exists():
            tensor_path = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii"

        img = nib.load(tensor_path).get_fdata().astype(np.float32)

        # Normalize
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = self._pad_or_crop_to_shape(img, self.target_shape)

        # Select channels if needed
        if self.input_channels < 3:
            if self.input_channels == 1:
                # Use just the first channel (or middle channel)
                img = img[1:2, :, :, :]  # Keep dimension
            elif self.input_channels == 2:
                # Use first two channels
                img = img[:2, :, :, :]

        # Convert to TorchIO Subject for augmentation
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.from_numpy(img).unsqueeze(0))
            )
            subject = self.transform(subject)
            img = subject.image.data.squeeze(0).numpy()

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
# TRAINING FUNCTION
# -----------------------
def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None, ema=None):
    model.train()
    train_losses = []
    train_preds = []
    train_trues = []
    
    for imgs, targets in tqdm(train_loader, desc="Training"):
        imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
        optimizer.step()
        
        # Update EMA
        if ema:
            ema.update()

        train_losses.append(loss.item())
        train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        train_trues.extend(targets.detach().cpu().numpy())

    return np.mean(train_losses), train_preds, train_trues

def validate_epoch(model, val_loader, device):
    model.eval()
    val_preds = []
    val_trues = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Validation"):
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(imgs)
            val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            val_trues.extend(targets.detach().cpu().numpy())
    
    return val_preds, val_trues

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print(f"🔄 Training {MODEL_TYPE} model with {INPUT_CHANNELS} channels...")

    with open(LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)

    pcr_data = {item['patient_id']: item for item in pcr_list}
    patient_ids = []
    labels = []

    print("🔎 Loading data...")

    for pid, info in pcr_data.items():
        pid = pid.strip()
        
        if 'pcr' in info and info['pcr'] in ["0", "1"]:
            tensor_path_niigz = DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
            tensor_path_nii = DATA_DIR / pid / f"{pid}_tensor_3ch.nii"

            if tensor_path_niigz.exists() or tensor_path_nii.exists():
                patient_ids.append(pid)
                labels.append(int(info['pcr']))

    print(f"✅ Data loaded: {len(patient_ids)} patients.")

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        patient_ids, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Enhanced datasets with augmentations
    train_dataset = MAMAMIA3DDatasetEnhanced(
        DATA_DIR, train_ids, train_labels, 
        augment=True, target_shape=TARGET_SHAPE, input_channels=INPUT_CHANNELS
    )
    val_dataset = MAMAMIA3DDatasetEnhanced(
        DATA_DIR, val_ids, val_labels, 
        augment=False, target_shape=TARGET_SHAPE, input_channels=INPUT_CHANNELS
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Get model
    model = get_model(MODEL_TYPE, INPUT_CHANNELS)
    model = model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Loss function with class weighting
    pos_weight = torch.tensor([len(train_labels) / sum(train_labels)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warmup
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=len(train_loader) * 20,  # 20 epochs for first cycle
        max_lr=LEARNING_RATE,
        min_lr=LEARNING_RATE * 0.01,
        warmup_steps=len(train_loader) * 2  # 2 epochs warmup
    )

    # EMA
    ema = None
    if USE_EMA:
        ema = EMA(model, decay=EMA_DECAY)
        ema.register()

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        train_loss, train_preds, train_trues = train_epoch(
            model, train_loader, criterion, optimizer, device, GRADIENT_CLIP, ema
        )
        
        # Validation (use EMA model if available)
        if ema:
            ema.apply_shadow()
        
        val_preds, val_trues = validate_epoch(model, val_loader, device)
        
        if ema:
            ema.restore()

        # Metrics
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_bal_acc = balanced_accuracy_score(val_trues, val_preds_binary)
        val_auc = roc_auc_score(val_trues, val_preds)
        
        train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
        train_bal_acc = balanced_accuracy_score(train_trues, train_preds_binary)
        train_auc = roc_auc_score(train_trues, train_preds)

        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Bal-Acc: {train_bal_acc:.4f}")
        print(f"  Val   - AUC: {val_auc:.4f}, Bal-Acc: {val_bal_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save EMA model if available
            save_model = model
            if ema:
                ema.apply_shadow()
                save_model = model
            torch.save(save_model.state_dict(), f'best_model_{MODEL_TYPE}_{INPUT_CHANNELS}ch.pth')
            if ema:
                ema.restore()
        else:
            patience_counter += 1

        # Update scheduler
        scheduler.step()

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    print(f"🏆 Best Validation AUC: {best_val_auc:.4f}")
    print("✅ Training completed.")