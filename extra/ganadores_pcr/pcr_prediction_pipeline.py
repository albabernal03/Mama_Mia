import os
import torch
import torch.nn as nn
import nibabel as nib
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
import torchmetrics

# Dataset
class PCRDataset(Dataset):
    def __init__(self, patient_ids, labels_df, data_dir):
        self.patient_ids = patient_ids
        self.labels_df = labels_df.set_index('patient_id')
        self.data_dir = data_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img_path = self.data_dir / f"{patient_id}.nii.gz"
        img = nib.load(str(img_path)).get_fdata()
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        label = int(self.labels_df.loc[patient_id]['pcr_response'])
        return img, label

# Modelo
class PCRModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        from medicalnet_models import resnet34
        model = resnet34(shortcut_type='B', sample_input_D=96, sample_input_H=96, sample_input_W=96, num_seg_classes=2)

        # Cargar pesos preentrenados
        pretrained_path = "C:/Users/usuario/Documents/Mama_Mia/mejoras/MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_34.pth"
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)

        self.backbone = model
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_auc = torchmetrics.classification.BinaryAUROC()

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        logits = self.head(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_auc(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

# Config
class Config:
    processed_data_dir = Path('D:/mama_mia_fixed_0000_0001_challenge_ready')
    labels_csv = Path('C:/Users/usuario/Documents/Mama_Mia/ganadores_pcr/pcr_labels.csv')
    batch_size = 4
    num_workers = 4
    cv_folds = 5
    lr = 1e-4
    max_epochs = 50

# Trainer
class Trainer:
    def __init__(self, config):
        self.config = config

    def train_model(self):
        labels_df = pd.read_csv(self.config.labels_csv)

        # Solo pacientes con label y archivo existente
        patients = []
        for pid in labels_df['patient_id']:
            img_path = self.config.processed_data_dir / f"{pid}.nii.gz"
            if img_path.exists():
                patients.append(pid)

        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        labels_for_split = labels_df.set_index('patient_id').loc[patients]['pcr_response']
        train_idx, val_idx = next(iter(skf.split(patients, labels_for_split)))

        train_patients = [patients[i] for i in train_idx]
        val_patients = [patients[i] for i in val_idx]

        train_ds = PCRDataset(train_patients, labels_df, self.config.processed_data_dir)
        val_ds = PCRDataset(val_patients, labels_df, self.config.processed_data_dir)

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        model = PCRModel(self.config)

        trainer = pl.Trainer(
            accelerator='gpu', devices=1, precision=16,
            max_epochs=self.config.max_epochs
        )

        trainer.fit(model, train_loader, val_loader)
        return model, trainer

# Main
if __name__ == "__main__":
    print("🎯 INICIANDO PIPELINE DE PREDICCIÓN pCR")
    config = Config()
    trainer = Trainer(config)
    model, pytorch_trainer = trainer.train_model()
