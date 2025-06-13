# training_clean.py
"""
Pipeline completo de entrenamiento MAMA-MIA con DATOS REALES de pCR
VERSIÓN FINAL LIMPIA - Con debug para KeyError
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def set_random_seeds(seed=42):
    """Fijar seeds para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MAMAMIADataset(Dataset):
    """Dataset optimizado para datos reales de pCR"""
    
    def __init__(self, 
                 data_dir: Path,
                 patient_ids: List[str],
                 labels: List[int],
                 transforms=None,
                 target_size=(96, 96, 48)):
        
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.labels = labels
        self.transforms = transforms
        self.target_size = target_size
        
        # Verificar archivos válidos
        self.valid_indices = self._check_valid_files()
        
        print(f"Dataset created: {len(self.valid_indices)}/{len(patient_ids)} valid patients")
        if len(self.labels) > 0:
            valid_labels = [self.labels[i] for i in self.valid_indices]
            pcr_count = sum(valid_labels)
            no_pcr_count = len(valid_labels) - pcr_count
            pcr_rate = pcr_count / len(valid_labels) if valid_labels else 0
            print(f"  pCR: {pcr_count} ({pcr_rate:.1%})")
            print(f"  No-pCR: {no_pcr_count} ({1-pcr_rate:.1%})")
    
    def _check_valid_files(self) -> List[int]:
        """Verificar que existen los archivos tensor"""
        valid_indices = []
        
        for i, patient_id in enumerate(self.patient_ids):
            tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
            if tensor_file.exists():
                try:
                    # Test de carga rápida
                    tensor_img = nib.load(tensor_file)
                    shape = tensor_img.shape
                    if len(shape) == 4 and shape[0] == 3:  # (3, H, W, D)
                        valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading {patient_id}: {e}")
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Mapear índice válido
        actual_idx = self.valid_indices[idx]
        patient_id = self.patient_ids[actual_idx]
        label = self.labels[actual_idx] if self.labels else 0
        
        # Cargar tensor 3-channel [pre, post, mask]
        tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
        tensor_img = nib.load(tensor_file)
        tensor_data = tensor_img.get_fdata().astype(np.float32)
        
        # Convertir a tensor PyTorch: (C, H, W, D)
        tensor = torch.from_numpy(tensor_data)
        
        # Padding uniforme
        tensor = self._pad_to_target_size(tensor)
        
        # Aplicar transformaciones
        if self.transforms:
            tensor = self.transforms(tensor)
        
        return {
            'tensor': tensor,
            'target': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id
        }
    
    def _pad_to_target_size(self, tensor):
        """Aplicar padding para llegar al tamaño target"""
        c, h, w, d = tensor.shape
        target_h, target_w, target_d = self.target_size
        
        # Padding
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_d = max(0, target_d - d)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            padding = (
                pad_d // 2, pad_d - pad_d // 2,
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2
            )
            tensor = F.pad(tensor, padding, mode='constant', value=0)
        
        # Center crop si es necesario
        c, h, w, d = tensor.shape
        if h > target_h or w > target_w or d > target_d:
            start_h = max(0, (h - target_h) // 2)
            start_w = max(0, (w - target_w) // 2)
            start_d = max(0, (d - target_d) // 2)
            
            tensor = tensor[:, 
                          start_h:start_h + target_h,
                          start_w:start_w + target_w,
                          start_d:start_d + target_d]
        
        return tensor

class DataAugmentation3D:
    """Augmentaciones optimizadas para datos reales"""
    
    def __init__(self, prob=0.6):
        self.prob = prob
    
    def __call__(self, tensor):
        # Random flip horizontal
        if torch.rand(1) < self.prob:
            tensor = torch.flip(tensor, [1])
        
        # Random flip sagital (menos probable)
        if torch.rand(1) < self.prob * 0.3:
            tensor = torch.flip(tensor, [2])
        
        # Intensity augmentation (solo pre y post, no mask)
        if torch.rand(1) < self.prob:
            factor = 0.85 + torch.rand(1) * 0.3  # 0.85 a 1.15
            tensor[0] = tensor[0] * factor  # Pre
            tensor[1] = tensor[1] * factor  # Post
        
        # Gaussian noise suave
        if torch.rand(1) < self.prob * 0.25:
            noise = torch.randn_like(tensor[:2]) * 0.005
            tensor[0] = tensor[0] + noise[0]
            tensor[1] = tensor[1] + noise[1]
        
        return tensor

class ResNet3DBlock(nn.Module):
    """Bloque ResNet 3D mejorado"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout1 = nn.Dropout3d(dropout)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.dropout2 = nn.Dropout3d(dropout)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout2(out)
        return out

class ResNet3D_MAMA_Real(nn.Module):
    """ResNet3D optimizado para predicción pCR con datos reales"""
    
    def __init__(self, in_channels=3, num_classes=2, dropout=0.4):
        super().__init__()
        
        # Stem mejorado
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        
        # Layers con dropout progresivo
        self.layer1 = self._make_layer(64, 64, 3, stride=1, dropout=dropout*0.4)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, dropout=dropout*0.6)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, dropout=dropout*0.8)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, dropout=dropout)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier más robusto para datos reales
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResNet3DBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Clasificación
        logits = self.classifier(x)
        return logits

class MAMAMIATrainer_Real:
    """Trainer para datos reales de pCR - VERSIÓN FINAL CON DEBUG"""
    
    def __init__(self, 
                 data_dir: Path,
                 splits_csv: Path,
                 pcr_labels_file: Path,
                 output_dir: Path,
                 config: Dict):
        
        self.data_dir = data_dir
        self.splits_csv = splits_csv
        self.pcr_labels_file = pcr_labels_file
        self.output_dir = output_dir
        self.config = config
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._setup_logging()
        
        # Cargar splits y labels reales
        self.splits_df = pd.read_csv(splits_csv)
        
        # CORREGIDO: Cargar datos reales de pCR (lista de diccionarios)
        with open(pcr_labels_file, 'r') as f:
            pcr_list = json.load(f)
        
        # Convertir lista a diccionario para acceso rápido por patient_id
        self.pcr_data = {}
        for item in pcr_list:
            patient_id = item['patient_id']
            self.pcr_data[patient_id] = item
        
        print(f"✅ Loaded {len(self.pcr_data)} patients with clinical data")
        
        # Preparar datos
        self.train_patients, self.train_labels = self._prepare_training_data_real()
        self.test_patients, self.test_labels = self._prepare_test_data_real()
        
        # Estadísticas
        self._print_real_statistics()
        
    def _setup_logging(self):
        log_file = self.output_dir / 'training_real_pcr.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _prepare_training_data_real(self):
        """Preparar datos de entrenamiento con labels REALES - CON DEBUG"""
        train_patients = self.splits_df['train_split'].dropna().tolist()
        train_labels = []
        valid_patients = []
        
        print(f"🔍 Processing {len(train_patients)} training patients...")
        
        for i, patient_id in enumerate(train_patients):
            if patient_id in self.pcr_data:
                try:
                    pcr_status = self.pcr_data[patient_id]['pcr']
                    if pcr_status != 'unknown':  # Excluir unknown
                        train_labels.append(int(pcr_status))
                        valid_patients.append(patient_id)
                    else:
                        print(f"⚠️  Patient {patient_id}: Unknown pCR status - excluded")
                except KeyError:
                    print(f"❌ ERROR: Patient {patient_id} missing 'pcr' field")
                    print(f"   Available keys: {list(self.pcr_data[patient_id].keys())}")
                    print(f"   Sample data: {str(self.pcr_data[patient_id])[:200]}...")
                    continue
            else:
                print(f"❌ Patient {patient_id} not found in clinical data")
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(train_patients)} patients...")
        
        print(f"✅ Training: {len(valid_patients)}/{len(train_patients)} patients with valid pCR data")
        return valid_patients, train_labels
    
    def _prepare_test_data_real(self):
        """Preparar datos de test con labels REALES - CON DEBUG"""
        test_patients = self.splits_df['test_split'].dropna().tolist()
        test_labels = []
        valid_patients = []
        
        print(f"🔍 Processing {len(test_patients)} test patients...")
        
        for i, patient_id in enumerate(test_patients):
            if patient_id in self.pcr_data:
                try:
                    pcr_status = self.pcr_data[patient_id]['pcr']
                    if pcr_status != 'unknown':
                        test_labels.append(int(pcr_status))
                        valid_patients.append(patient_id)
                    else:
                        print(f"⚠️  Test patient {patient_id}: Unknown pCR status - excluded")
                except KeyError:
                    print(f"❌ ERROR: Test patient {patient_id} missing 'pcr' field")
                    print(f"   Available keys: {list(self.pcr_data[patient_id].keys())}")
                    continue
            else:
                print(f"❌ Test patient {patient_id} not found in clinical data")
        
        print(f"✅ Test: {len(valid_patients)}/{len(test_patients)} patients with valid pCR data")
        return valid_patients, test_labels
    
    def _print_real_statistics(self):
        """Estadísticas de datos reales"""
        train_pcr_rate = sum(self.train_labels) / len(self.train_labels) if self.train_labels else 0
        test_pcr_rate = sum(self.test_labels) / len(self.test_labels) if self.test_labels else 0
        
        print("\n=== REAL pCR STATISTICS ===")
        print(f"Training set:")
        print(f"  Total patients: {len(self.train_patients)}")
        print(f"  pCR patients: {sum(self.train_labels)}")
        print(f"  No-pCR patients: {len(self.train_labels) - sum(self.train_labels)}")
        print(f"  pCR rate: {train_pcr_rate:.1%}")
        
        print(f"Test set:")
        print(f"  Total patients: {len(self.test_patients)}")
        print(f"  pCR patients: {sum(self.test_labels)}")
        print(f"  No-pCR patients: {len(self.test_labels) - sum(self.test_labels)}")
        print(f"  pCR rate: {test_pcr_rate:.1%}")
        
        # Balance check
        if train_pcr_rate < 0.15 or train_pcr_rate > 0.85:
            print(f"⚠️  IMBALANCED DATASET: pCR rate = {train_pcr_rate:.1%}")
    
    def create_datasets(self, train_ids, val_ids, train_labels, val_labels):
        """Crear datasets"""
        train_transforms = DataAugmentation3D(prob=0.7)
        
        train_dataset = MAMAMIADataset(
            data_dir=self.data_dir,
            patient_ids=train_ids,
            labels=train_labels,
            transforms=train_transforms
        )
        
        val_dataset = MAMAMIADataset(
            data_dir=self.data_dir,
            patient_ids=val_ids,
            labels=val_labels,
            transforms=None
        )
        
        return train_dataset, val_dataset
    
    def train_fold(self, fold, train_dataset, val_dataset):
        """Entrenar un fold"""
        print(f"🚀 Training fold {fold}")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Modelo
        model = ResNet3D_MAMA_Real(
            in_channels=3,
            num_classes=2,
            dropout=self.config['dropout']
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, verbose=True
        )
        
        # Loss con class weights para balance
        train_labels = [train_dataset.labels[i] for i in train_dataset.valid_indices]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_val_auc = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            train_pbar = tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}")
            for batch in train_pbar:
                tensors = batch['tensor'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                outputs = model(tensors)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
                train_targets.extend(targets.cpu().numpy())
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    tensors = batch['tensor'].to(device)
                    targets = batch['target'].to(device)
                    
                    outputs = model(tensors)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Metrics
            train_auc = roc_auc_score(train_targets, train_preds) if len(set(train_targets)) > 1 else 0.5
            val_auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.5
            
            train_acc = accuracy_score(train_targets, [1 if p > 0.5 else 0 for p in train_preds])
            val_acc = accuracy_score(val_targets, [1 if p > 0.5 else 0 for p in val_preds])
            
            # Scheduler
            scheduler.step(val_auc)
            
            print(
                f"Fold {fold}, Epoch {epoch+1}: "
                f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'config': self.config
                }, self.output_dir / f'best_model_fold_{fold}_REAL.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_auc
    
    def run_cross_validation(self):
        """5-fold CV con datos reales"""
        if len(self.train_patients) == 0:
            print("❌ No patients available for training!")
            return 0.0, 0.0, []
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.train_patients, self.train_labels)):
            print(f"\n🚀 Starting fold {fold}")
            
            fold_train_patients = [self.train_patients[i] for i in train_idx]
            fold_val_patients = [self.train_patients[i] for i in val_idx]
            fold_train_labels = [self.train_labels[i] for i in train_idx]
            fold_val_labels = [self.train_labels[i] for i in val_idx]
            
            train_dataset, val_dataset = self.create_datasets(
                fold_train_patients, fold_val_patients, 
                fold_train_labels, fold_val_labels
            )
            
            fold_auc = self.train_fold(fold, train_dataset, val_dataset)
            fold_aucs.append(fold_auc)
            
            torch.cuda.empty_cache()
        
        # Resultados
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        print("\n=== FINAL RESULTS WITH REAL DATA ===")
        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {fold_aucs}")
        
        # Guardar resultados
        results = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_aucs': fold_aucs,
            'config': self.config,
            'data_type': 'REAL_PCR_LABELS'
        }
        
        with open(self.output_dir / 'cv_results_REAL.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return mean_auc, std_auc, fold_aucs

def main():
    """Entrenar con datos REALES de pCR - VERSIÓN FINAL LIMPIA"""
    
    set_random_seeds(42)
    
    # Configuración optimizada para datos reales
    config = {
        'batch_size': 6,
        'learning_rate': 3e-5,
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 15,
        'dropout': 0.4
    }
    
    # Paths
    data_dir = Path("D:/mama_mia_final_corrected")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    pcr_labels_file = Path("D:/clinical_data_complete.json")
    output_dir = Path("D:/mama_mia_REAL_PCR_results")
    
    print("🎉 TRAINING WITH REAL pCR DATA - FINAL CLEAN VERSION")
    print(f"Data directory: {data_dir}")
    print(f"Real pCR labels: {pcr_labels_file}")
    print(f"Output: {output_dir}")
    print(f"Config: {config}")
    
    # Verificar archivos
    if not pcr_labels_file.exists():
        print(f"ERROR: pCR labels file not found: {pcr_labels_file}")
        return
    
    # Entrenar
    trainer = MAMAMIATrainer_Real(data_dir, splits_csv, pcr_labels_file, output_dir, config)
    
    print("\n🚀 Starting training with REAL pCR labels...")
    print("Expected AUC range: 0.65-0.85 (vs 0.56 with mock data)")
    
    mean_auc, std_auc, fold_aucs = trainer.run_cross_validation()
    
    print(f"\n🎯 FINAL RESULTS WITH REAL DATA:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Individual fold AUCs: {fold_aucs}")
    
    if mean_auc > 0.75:
        print("🎉 EXCELLENT: AUC > 0.75 - Ready for publication!")
    elif mean_auc > 0.65:
        print("✅ GOOD: AUC > 0.65 - Clinical relevance achieved")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Consider Swin Transformer architecture")
    
    print(f"\nModels saved in: {output_dir}")

if __name__ == "__main__":
    main()