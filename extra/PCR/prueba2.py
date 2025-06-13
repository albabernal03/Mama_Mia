import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from sklearn.metrics import roc_auc_score
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

# ---------------------------------------------------
# MODELO 3D RESNET34
# ---------------------------------------------------

class ResNet3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D34(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet3D34, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels))
        layers = [ResNet3DBlock(in_channels, out_channels, stride, downsample)]
        layers += [ResNet3DBlock(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class MamaMiaPaperModel(nn.Module):
    def __init__(self, clinical_features=0):
        super(MamaMiaPaperModel, self).__init__()
        self.pre_backbone = ResNet3D34()
        self.post_backbone = ResNet3D34()
        combined_features = 512 * 2 + clinical_features
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.head_pcr = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, pre_image, post_image, clinical_data=None):
        pre_features = self.pre_backbone(pre_image)
        post_features = self.post_backbone(post_image)
        combined = torch.cat([pre_features, post_features], dim=1)
        if clinical_data is not None:
            combined = torch.cat([combined, clinical_data], dim=1)
        shared_features = self.fusion_layer(combined)
        head_pcr_out = torch.sigmoid(self.head_pcr(shared_features))
        return head_pcr_out

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

def resize_volume(img, target_shape=(64, 64, 32)):
    img = torch.FloatTensor(img).unsqueeze(0)
    img = F.interpolate(img.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False)
    return img.squeeze(0)

class MamaMiaDataset(Dataset):
    def __init__(self, patient_ids, labels, cropped_data_dir, split_type):
        self.patient_ids = patient_ids
        self.labels = labels
        self.cropped_data_dir = Path(cropped_data_dir)
        self.split_type = split_type

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        patient_dir = self.cropped_data_dir / "images" / self.split_type / patient_id
        seg_dir = self.cropped_data_dir / "segmentations" / self.split_type / patient_id
        img_files = list(patient_dir.glob("*_cropped.nii.gz"))
        seg_files = list(seg_dir.glob("*_seg_cropped.nii.gz"))
        mask = nib.load(seg_files[0]).get_fdata().astype(np.float32)
        numbered_files = {}
        for img_file in img_files:
            parts = img_file.name.replace('_cropped.nii.gz', '').split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                numbered_files[int(parts[-1])] = img_file
        pre_image = nib.load(numbered_files[0]).get_fdata().astype(np.float32)
        post_image = nib.load(numbered_files[1]).get_fdata().astype(np.float32)
        pre_tensor = resize_volume(pre_image)
        post_tensor = resize_volume(post_image)
        return pre_tensor, post_tensor, torch.FloatTensor([label])

# ---------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------

def train_model(cropped_data_dir, splits_csv, pcr_labels_csv, epochs=50, batch_size=8, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    splits_df = pd.read_csv(splits_csv)
    pcr_df = pd.read_csv(pcr_labels_csv)

    pcr_dict = dict(zip(pcr_df['patient_id'], pcr_df[pcr_df.columns[1]]))

    train_patients = []
    train_labels = []
    for p in splits_df['train_split'].dropna().tolist():
        if p in pcr_dict:
            train_patients.append(p)
            train_labels.append(pcr_dict[p])

    test_patients = []
    test_labels = []
    for p in splits_df['test_split'].dropna().tolist():
        if p in pcr_dict:
            test_patients.append(p)
            test_labels.append(pcr_dict[p])

    train_dataset = MamaMiaDataset(train_patients, train_labels, cropped_data_dir, 'train_split')
    test_dataset = MamaMiaDataset(test_patients, test_labels, cropped_data_dir, 'test_split')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MamaMiaPaperModel()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for pre_images, post_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pre_images, post_images, labels = pre_images.to(device), post_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

        model.eval()
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for pre_images, post_images, labels in test_loader:
                pre_images, post_images = pre_images.to(device), post_images.to(device)
                outputs = model(pre_images, post_images)
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
        auc = roc_auc_score(all_labels, all_outputs)
        print(f"Validation AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_paper_model.pth')

    print(f"Best Validation AUC: {best_auc:.4f}")

# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MAMA-MIA CNN Paper Pipeline')
    parser.add_argument('--cropped_data_dir', type=str, required=True, help='Path to cropped data')
    parser.add_argument('--splits_csv', type=str, required=True, help='Path to train/test splits CSV')
    parser.add_argument('--pcr_labels_csv', type=str, required=True, help='Path to PCR labels CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    train_model(
        cropped_data_dir=args.cropped_data_dir,
        splits_csv=args.splits_csv,
        pcr_labels_csv=args.pcr_labels_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()