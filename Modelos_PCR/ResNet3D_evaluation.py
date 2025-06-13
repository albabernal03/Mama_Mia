# test_evaluation_only.py
"""
Evaluación SOLO en test set usando modelos ya entrenados
NO re-entrena - usa los modelos guardados en mama_mia_REAL_PCR_results
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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Usar las mismas clases del entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MAMAMIADataset(Dataset):
    """Dataset optimizado para datos reales de pCR"""
    
    def __init__(self, 
                 data_dir: Path,
                 patient_ids: list,
                 labels: list,
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
    
    def _check_valid_files(self) -> list:
        """Verificar que existen los archivos tensor"""
        valid_indices = []
        
        for i, patient_id in enumerate(self.patient_ids):
            tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
            if tensor_file.exists():
                try:
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

def prepare_test_data():
    """Preparar datos de test con labels reales"""
    # Paths
    data_dir = Path("D:/mama_mia_final_corrected")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    pcr_labels_file = Path("D:/clinical_data_complete.json")
    
    # Cargar splits
    splits_df = pd.read_csv(splits_csv)
    
    # Cargar datos de pCR
    with open(pcr_labels_file, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {}
    for item in pcr_list:
        pcr_data[item['patient_id']] = item
    
    # Preparar test set
    test_patients = splits_df['test_split'].dropna().tolist()
    test_labels = []
    valid_patients = []
    
    print(f"🔍 Processing {len(test_patients)} test patients...")
    
    for patient_id in test_patients:
        if patient_id in pcr_data:
            try:
                pcr_status = pcr_data[patient_id]['pcr']
                if pcr_status != 'unknown':
                    test_labels.append(int(pcr_status))
                    valid_patients.append(patient_id)
            except KeyError:
                print(f"❌ Patient {patient_id} missing 'pcr' field")
                continue
    
    print(f"✅ Test set: {len(valid_patients)} patients with valid pCR data")
    return data_dir, valid_patients, test_labels

def evaluate_on_test():
    """Evaluación final usando modelos ya entrenados"""
    print("🧪 EVALUATING ON INDEPENDENT TEST SET...")
    print("📁 Using pre-trained models from: mama_mia_REAL_PCR_results")
    
    # Configuración (debe coincidir con el entrenamiento)
    config = {
        'batch_size': 6,
        'dropout': 0.4
    }
    
    # Preparar datos de test
    data_dir, test_patients, test_labels = prepare_test_data()
    models_dir = Path("D:/mama_mia_REAL_PCR_results")
    
    # Crear dataset de test
    test_dataset = MAMAMIADataset(
        data_dir=data_dir,
        patient_ids=test_patients,
        labels=test_labels,
        transforms=None  # Sin augmentación en test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Cargar y evaluar cada modelo
    test_predictions = []
    
    for fold in range(5):
        model_path = models_dir / f'best_model_fold_{fold}_REAL.pth'
        
        if not model_path.exists():
            print(f"⚠️ Model fold {fold} not found at {model_path}")
            continue
            
        print(f"📂 Loading model fold {fold}...")
        
        # Cargar modelo
        model = ResNet3D_MAMA_Real(
            in_channels=3,
            num_classes=2,
            dropout=config['dropout']
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Predicciones
        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                tensors = batch['tensor'].to(device)
                outputs = model(tensors)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                fold_preds.extend(probs)
        
        test_predictions.append(fold_preds)
        print(f"✅ Fold {fold} predictions collected ({len(fold_preds)} samples)")
    
    # Ensemble promedio
    if test_predictions:
        ensemble_preds = np.mean(test_predictions, axis=0)
        test_targets = [test_dataset.labels[i] for i in test_dataset.valid_indices]
        
        # Calcular métricas finales
        test_auc = roc_auc_score(test_targets, ensemble_preds)
        test_acc = accuracy_score(test_targets, [1 if p > 0.5 else 0 for p in ensemble_preds])
        
        print(f"\n🎯 FINAL TEST RESULTS:")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test pCR Rate: {np.mean(test_targets):.1%}")
        print(f"Test Set Size: {len(test_targets)} patients")
        print(f"Models Used: {len(test_predictions)}/5 folds")
        
        # Cargar resultados de CV para comparar
        cv_results_file = models_dir / 'cv_results_REAL.json'
        if cv_results_file.exists():
            with open(cv_results_file, 'r') as f:
                cv_results = json.load(f)
            
            cv_auc = cv_results['mean_auc']
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"Cross-Validation AUC: {cv_auc:.4f}")
            print(f"Independent Test AUC: {test_auc:.4f}")
            
            performance_gap = cv_auc - test_auc
            if performance_gap > 0.05:
                print(f"⚠️ POTENTIAL OVERFITTING: CV-Test gap = {performance_gap:.3f}")
            elif test_auc > cv_auc:
                print("✅ EXCELLENT GENERALIZATION: Test > CV performance")
            else:
                print("✅ CONSISTENT PERFORMANCE: Similar CV and Test")
        
        # Guardar resultados de test
        test_results = {
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'test_predictions': ensemble_preds.tolist(),
            'test_targets': test_targets,
            'num_models_ensemble': len(test_predictions),
            'test_set_size': len(test_targets),
            'patient_ids': [test_dataset.patient_ids[i] for i in test_dataset.valid_indices]
        }
        
        results_file = models_dir / 'test_results_FINAL.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Interpretación final
        print(f"\n🏆 FINAL INTERPRETATION:")
        if test_auc > 0.75:
            print("🎉 EXCELLENT: Test AUC > 0.75 - Publication ready!")
        elif test_auc > 0.65:
            print("✅ GOOD: Test AUC > 0.65 - Clinically relevant") 
        else:
            print("⚠️ NEEDS IMPROVEMENT: Test AUC < 0.65 - Consider architecture changes")
        
        return test_auc, test_acc
    else:
        print("❌ No models found for test evaluation!")
        return 0.0, 0.0

if __name__ == "__main__":
    print("🚀 EVALUATING PRE-TRAINED MODELS ON TEST SET")
    print("=" * 60)
    
    test_auc, test_acc = evaluate_on_test()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 60)