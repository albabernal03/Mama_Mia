# evaluate_resnet3d_test.py
"""
Evaluación del modelo ResNet3D TorchVision en test set independiente
"""

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report
import pandas as pd
from pathlib import Path
import json
import torchvision.models.video as models3d

# Configuración
DATA_DIR = Path(r"D:\mama_mia_final_corrected")
LABELS_FILE = Path(r"D:\clinical_data_complete.json")
SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
MODEL_PATH = "best_model_3dcnn_resnet.pth"
BATCH_SIZE = 8
TARGET_SHAPE = (128, 128, 128)

class ResNet3D18(nn.Module):
    """Mismo modelo que usaste para entrenar"""
    def __init__(self):
        super(ResNet3D18, self).__init__()
        self.backbone = models3d.r3d_18(pretrained=False)
        # Adaptamos input 3 canales
        self.backbone.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # Adaptamos output a binario
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)

class MAMAMIA3DDataset(Dataset):
    """Mismo dataset que usaste para entrenar"""
    def __init__(self, data_dir, patient_ids, labels, target_shape=(128, 128, 128)):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.labels = labels
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

def prepare_test_data():
    """Preparar datos de test usando splits oficiales"""
    print("🔍 Preparing official test set...")
    
    # Cargar splits oficiales
    splits_df = pd.read_csv(SPLITS_CSV)
    
    # Cargar datos de pCR
    with open(LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    
    # Obtener test patients oficiales
    test_patients = splits_df['test_split'].dropna().tolist()
    test_labels = []
    valid_patients = []
    
    print(f"🔍 Processing {len(test_patients)} official test patients...")
    
    for patient_id in test_patients:
        patient_id = patient_id.strip()
        
        if patient_id in pcr_data:
            if 'pcr' in pcr_data[patient_id] and pcr_data[patient_id]['pcr'] in ["0", "1"]:
                # Verificar que existe el tensor
                tensor_path_niigz = DATA_DIR / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
                tensor_path_nii = DATA_DIR / patient_id / f"{patient_id}_tensor_3ch.nii"
                
                if tensor_path_niigz.exists() or tensor_path_nii.exists():
                    valid_patients.append(patient_id)
                    test_labels.append(int(pcr_data[patient_id]['pcr']))
    
    print(f"✅ Official test set: {len(valid_patients)} patients with valid data")
    print(f"   pCR rate: {np.mean(test_labels):.1%}")
    
    return valid_patients, test_labels

def evaluate_model():
    """Evaluar modelo entrenado en test set oficial"""
    print("🧪 EVALUATING ResNet3D TorchVision ON OFFICIAL TEST SET")
    print("=" * 60)
    
    # Preparar datos de test
    test_patients, test_labels = prepare_test_data()
    
    # Crear dataset de test
    test_dataset = MAMAMIA3DDataset(
        data_dir=DATA_DIR,
        patient_ids=test_patients,
        labels=test_labels,
        target_shape=TARGET_SHAPE
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet3D18()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {MODEL_PATH}")
    
    # Evaluación
    test_preds = []
    test_targets = []
    
    print("🔄 Running inference...")
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).squeeze()
            
            test_preds.extend(probs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Convertir a arrays
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    # Calcular métricas
    test_auc = roc_auc_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, test_preds_binary)
    test_bal_acc = balanced_accuracy_score(test_targets, test_preds_binary)
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Test Set Size: {len(test_targets)} patients")
    print(f"Test pCR Rate: {np.mean(test_targets):.1%}")
    
    # Reporte detallado
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(test_targets, test_preds_binary, 
                              target_names=['No-pCR', 'pCR']))
    
    # Comparación con validation
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Validation Balanced Accuracy: 0.6101")  # Tu resultado
    print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
    
    performance_gap = 0.6101 - test_bal_acc
    if performance_gap > 0.05:
        print(f"⚠️ POTENTIAL OVERFITTING: Val-Test gap = {performance_gap:.3f}")
    elif test_bal_acc > 0.6101:
        print("✅ EXCELLENT GENERALIZATION: Test > Validation")
    else:
        print("✅ CONSISTENT PERFORMANCE: Similar Validation and Test")
    
    # Guardar resultados
    results = {
        'test_auc': float(test_auc),
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'test_predictions': test_preds.tolist(),
        'test_targets': test_targets.tolist(),
        'test_set_size': len(test_targets),
        'validation_balanced_accuracy': 0.6101,
        'patient_ids': test_patients
    }
    
    with open('resnet3d_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: resnet3d_test_results.json")
    
    # Interpretación final
    print(f"\n🏆 FINAL INTERPRETATION:")
    if test_auc > 0.75:
        print("🎉 EXCELLENT: Test AUC > 0.75 - Publication ready!")
    elif test_auc > 0.65:
        print("✅ GOOD: Test AUC > 0.65 - Clinically relevant") 
    elif test_auc > 0.60:
        print("✓ ACCEPTABLE: Test AUC > 0.60 - Moderate performance")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Test AUC < 0.60")
    
    return test_auc, test_acc, test_bal_acc

if __name__ == "__main__":
    # Verificar que existe el modelo
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please make sure you have trained and saved the model first.")
        exit(1)
    
    test_auc, test_acc, test_bal_acc = evaluate_model()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
    print("=" * 60)