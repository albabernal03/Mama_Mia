# pipeline_cnn_split.py
# CNN Pipeline con separación explícita de train/val/test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# IMPORTACIONES NECESARIAS (suponemos que están definidas en otro script o este archivo):
from model import MamaMiaPaperModel, FocalLoss
from dataset import MamaMiaDataset  # dataset definido igual que antes

class PaperPipelineTrainer:
    def __init__(self, cropped_data_dir, splits_csv, pcr_labels_csv, clinical_data_json=None):
        self.cropped_data_dir = cropped_data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")

        # Load data
        self.splits_df = pd.read_csv(splits_csv)
        self.pcr_df = pd.read_csv(pcr_labels_csv)

        # Clinical data (opcional)
        self.clinical_data = None
        if clinical_data_json and Path(clinical_data_json).exists():
            with open(clinical_data_json, 'r') as f:
                self.clinical_data = json.load(f)
            print(f"✅ Loaded clinical data for fusion")

        # Preparar pacientes y etiquetas
        all_train_patients, all_train_labels = self._extract_split('train_split')
        self.test_patients, self.test_labels = self._extract_split('test_split')

        # Dividir en train + val (80/20 estratificado)
        self.train_patients, self.val_patients, self.train_labels, self.val_labels = train_test_split(
            all_train_patients, all_train_labels, test_size=0.2, stratify=all_train_labels, random_state=42
        )

        print(f"🌟 TRAINING SETUP")
        print(f"  Train: {len(self.train_patients)} pacientes")
        print(f"  Val:   {len(self.val_patients)} pacientes")
        print(f"  Test:  {len(self.test_patients)} pacientes")

    def _extract_split(self, split_name):
        if split_name not in self.splits_df.columns:
            return [], []

        patients = self.splits_df[split_name].dropna().tolist()
        labels = []

        pcr_col = next((col for col in ['pcr', 'PCR', 'pcr_response', 'PCR_response'] if col in self.pcr_df.columns), None)
        if pcr_col is None:
            raise ValueError("No se encontró columna de pCR en CSV")

        label_dict = dict(zip(self.pcr_df['patient_id'], self.pcr_df[pcr_col]))

        valid_patients = []
        for p in patients:
            if p in label_dict and label_dict[p] in [0, 1]:
                valid_patients.append(p)
                labels.append(int(label_dict[p]))

        return valid_patients, labels

    def create_datasets(self):
        train_dataset = MamaMiaDataset(self.train_patients, self.train_labels, self.cropped_data_dir, 'train_split')
        val_dataset   = MamaMiaDataset(self.val_patients, self.val_labels, self.cropped_data_dir, 'train_split')
        test_dataset  = MamaMiaDataset(self.test_patients, self.test_labels, self.cropped_data_dir, 'test_split')
        return train_dataset, val_dataset, test_dataset

    def evaluate_model(self, model, data_loader):
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                pre_images = batch['pre_image'].to(self.device)
                post_images = batch['post_image'].to(self.device)
                labels = batch['label'].to(self.device)

                _, out, _ = model(pre_images, post_images)
                preds = out.cpu().numpy().flatten()
                targets = labels.cpu().numpy().flatten()

                all_predictions.extend(preds)
                all_labels.extend(targets)

        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except:
            auc = 0.5

        return auc

    def train_model(self, num_epochs=30, batch_size=4, learning_rate=1e-4):
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MamaMiaPaperModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        criterion_type = nn.BCELoss()
        criterion_pcr = FocalLoss()

        best_val_auc = 0

        for epoch in range(num_epochs):
            model.train()
            losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                pre = batch['pre_image'].to(self.device)
                post = batch['post_image'].to(self.device)
                label = batch['label'].to(self.device)

                out_type, out_pcr, _ = model(pre, post)
                loss = criterion_type(out_type, label) + criterion_pcr(out_pcr, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            val_auc = self.evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {np.mean(losses):.4f} | Val AUC: {val_auc:.4f}")
            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'best_model_val.pth')

        print("\nBest validation AUC:", best_val_auc)

        model.load_state_dict(torch.load('best_model_val.pth'))
        test_auc = self.evaluate_model(model, test_loader)
        print("Final Test AUC:", test_auc)

        return model

if __name__ == '__main__':
    cropped_data_dir = "C:/Users/usuario/Documents/Mama_Mia/cropped_data"
    splits_csv = "C:/Users/usuario/Documents/Mama_Mia/datos/train_test_splits.csv"
    pcr_labels_csv = "C:/Users/usuario/Documents/Mama_Mia/PCR/pcr_labels.csv"

    trainer = PaperPipelineTrainer(cropped_data_dir, splits_csv, pcr_labels_csv)
    trainer.train_model()
