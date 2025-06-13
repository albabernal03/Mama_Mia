# fixed_evaluation_with_predictions.py
"""
Evaluación CORREGIDA con tus predicciones de segmentación
- Usa datos originales correctos
- Solo PCR 0/1 (sin otros datos clínicos)
- Modelos incluidos directamente
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
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODELOS INCLUIDOS DIRECTAMENTE (sin importaciones externas)
# =============================================================================

class UltraFastCNN3D(nn.Module):
    """CNN 3D ultra-rápido - INCLUIDO DIRECTAMENTE"""
    
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 2
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FastResNet3D(nn.Module):
    """ResNet 3D simple - INCLUIDO DIRECTAMENTE"""
    
    def __init__(self, in_channels=3, num_classes=2, model_size='small'):
        super().__init__()
        
        if model_size == 'tiny':
            channels = [16, 32, 64, 128]
            blocks = [1, 1, 1, 1]
        elif model_size == 'small':
            channels = [32, 64, 128, 256]
            blocks = [2, 2, 2, 2]
        else:  # medium
            channels = [64, 128, 256, 512]
            blocks = [2, 2, 3, 2]
        
        # Entrada
        self.conv1 = nn.Conv3d(in_channels, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        
        # Layers ResNet
        self.layer1 = self._make_layer(channels[0], channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(channels[3], num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # First block with potential stride
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels)
            ))
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x) + x  # Residual connection
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# =============================================================================
# DATASET CORREGIDO PARA DATOS ORIGINALES
# =============================================================================

class CorrectedTestDataset(Dataset):
    """Dataset que usa los datos originales + predicciones de segmentación"""
    
    def __init__(self, 
                 images_dir: Path,  # C:\Users\usuario\Documents\Mama_Mia\datos\images
                 predictions_dir: Path,  # C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output
                 patient_ids: List[str],
                 target_size=(48, 48, 24),
                 use_channels='all'):
        
        self.images_dir = Path(images_dir)
        self.predictions_dir = Path(predictions_dir)
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.use_channels = use_channels
        
        # Verificar archivos disponibles
        self.valid_indices = []
        self.available_data = []
        
        print(f"🔍 Buscando datos en:")
        print(f"   📁 Imágenes: {self.images_dir}")
        print(f"   🔮 Predicciones: {self.predictions_dir}")
        
        for i, pid in enumerate(patient_ids):
            # Buscar imágenes originales en formato estándar MAMA-MIA
            pre_file = self._find_image_file(pid, 'pre')
            post_file = self._find_image_file(pid, 'post')
            predicted_mask = self._find_predicted_mask(pid)
            
            if pre_file and post_file and predicted_mask:
                self.valid_indices.append(i)
                self.available_data.append({
                    'patient_id': pid,
                    'pre_file': pre_file,
                    'post_file': post_file,
                    'predicted_mask_file': predicted_mask
                })
                
        print(f"📊 Dataset válido: {len(self.valid_indices)} pacientes de {len(patient_ids)}")
        
        if len(self.valid_indices) == 0:
            print("⚠️ No se encontraron pacientes válidos!")
            print("🔍 Verificando estructura de directorios...")
            self._debug_directory_structure()
    
    def _find_image_file(self, patient_id: str, phase: str) -> Optional[Path]:
        """Buscar archivo de imagen (pre o post contraste)"""
        
        # Patrones comunes para archivos de imagen
        phase_patterns = {
            'pre': ['_0000.nii.gz', '_pre.nii.gz', '_T1.nii.gz', 'pre.nii.gz'],
            'post': ['_0001.nii.gz', '_post.nii.gz', '_T1c.nii.gz', 'post.nii.gz', '.nii.gz']
        }
        
        # Directorios donde buscar
        search_dirs = [
            self.images_dir / patient_id,
            self.images_dir / "images" / patient_id,
            self.images_dir / "train" / patient_id,
            self.images_dir / "test" / patient_id,
            self.images_dir
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in phase_patterns[phase]:
                    # Buscar con ID del paciente + patrón
                    file_path = search_dir / f"{patient_id}{pattern}"
                    if file_path.exists():
                        return file_path
                
                # Buscar cualquier archivo que contenga el ID y el patrón
                for nii_file in search_dir.glob("*.nii.gz"):
                    filename = nii_file.name.lower()
                    if patient_id.lower() in filename:
                        for pattern in phase_patterns[phase]:
                            if pattern.replace('.nii.gz', '') in filename:
                                return nii_file
        
        return None
    
    def _find_predicted_mask(self, patient_id: str) -> Optional[Path]:
        """Buscar predicción de segmentación"""
        
        # Patrones para predicciones
        possible_patterns = [
            f"{patient_id}.nii.gz",
            f"{patient_id}_pred.nii.gz",
            f"{patient_id}_prediction.nii.gz", 
            f"{patient_id}_seg.nii.gz",
            f"pred_{patient_id}.nii.gz",
            f"prediction_{patient_id}.nii.gz"
        ]
        
        # Buscar en directorios de predicciones
        search_dirs = [
            self.predictions_dir,
            self.predictions_dir / "predictions",
            self.predictions_dir / "segmentations", 
            self.predictions_dir / "results",
            self.predictions_dir / patient_id
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in possible_patterns:
                    pred_file = search_dir / pattern
                    if pred_file.exists():
                        return pred_file
                
                # Buscar cualquier archivo que contenga el ID
                for nii_file in search_dir.glob("**/*.nii.gz"):
                    if patient_id in nii_file.name:
                        return nii_file
        
        return None
    
    def _debug_directory_structure(self):
        """Debug para ver estructura de directorios"""
        print(f"\n🔍 DEBUG - Estructura de directorios:")
        
        if self.images_dir.exists():
            print(f"📁 {self.images_dir}:")
            subdirs = [d for d in self.images_dir.iterdir() if d.is_dir()]
            files = [f for f in self.images_dir.iterdir() if f.is_file() and f.suffix == '.gz']
            print(f"   📂 Subdirectorios: {len(subdirs)} (ej: {subdirs[:3] if subdirs else 'ninguno'})")
            print(f"   📄 Archivos .nii.gz: {len(files)} (ej: {[f.name for f in files[:3]][:3] if files else 'ninguno'})")
        
        if self.predictions_dir.exists():
            print(f"🔮 {self.predictions_dir}:")
            pred_files = list(self.predictions_dir.glob("**/*.nii.gz"))
            print(f"   📄 Predicciones encontradas: {len(pred_files)}")
            if pred_files:
                print(f"   📝 Ejemplos: {[f.name for f in pred_files[:5]]}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_info = self.available_data[idx]
        patient_id = data_info['patient_id']
        
        try:
            # Cargar imágenes originales
            pre_data = nib.load(data_info['pre_file']).get_fdata().astype(np.float32)
            post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
            
            # Cargar predicción de segmentación
            predicted_mask_data = nib.load(data_info['predicted_mask_file']).get_fdata().astype(np.float32)
            
            # Procesar máscara predicha
            if predicted_mask_data.ndim > 3:
                if predicted_mask_data.shape[0] == 2:  # [background, tumor]
                    predicted_mask_data = predicted_mask_data[1]
                elif predicted_mask_data.shape[-1] == 2:  # [H, W, D, classes]
                    predicted_mask_data = predicted_mask_data[..., 1]
                else:
                    predicted_mask_data = predicted_mask_data[0] if predicted_mask_data.shape[0] > 1 else predicted_mask_data.squeeze()
            
            # Normalización simple
            if np.max(pre_data) > 0:
                pre_data = pre_data / np.percentile(pre_data[pre_data > 0], 99)
            if np.max(post_data) > 0:
                post_data = post_data / np.percentile(post_data[post_data > 0], 99)
            
            # Binarizar máscara predicha
            predicted_mask_binary = (predicted_mask_data > 0.5).astype(np.float32)
            
            # Crear tensor según configuración de canales
            if self.use_channels == 'pre_only':
                selected_data = np.stack([pre_data], axis=0)
            elif self.use_channels == 'post_only':
                selected_data = np.stack([post_data], axis=0)
            elif self.use_channels == 'pre_post':
                selected_data = np.stack([pre_data, post_data], axis=0)
            else:  # 'all' - FORMATO: [pre, post, predicted_mask]
                selected_data = np.stack([pre_data, post_data, predicted_mask_binary], axis=0)
            
            # Resize a tamaño objetivo
            tensor = torch.from_numpy(selected_data)
            tensor = F.interpolate(
                tensor.unsqueeze(0), 
                size=self.target_size, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
            
            return {
                'tensor': tensor.float(),
                'patient_id': patient_id,
                'mask_source': 'predicted'
            }
            
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {e}")
            n_channels = {'pre_only': 1, 'post_only': 1, 'pre_post': 2, 'all': 3}[self.use_channels]
            dummy_tensor = torch.zeros((n_channels, *self.target_size))
            return {
                'tensor': dummy_tensor,
                'patient_id': patient_id,
                'mask_source': 'dummy'
            }

# =============================================================================
# EVALUADOR CORREGIDO
# =============================================================================

class CorrectedModelEvaluator:
    """Evaluador corregido que carga modelos sin importaciones externas"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = []
        self.config = None
        
        self._load_models()
    
    def _load_models(self):
        """Cargar modelos del ensemble"""
        
        # Cargar configuración
        config_file = self.model_dir / 'final_results.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                results = json.load(f)
                self.config = results['best_config']
                print(f"✅ Configuración cargada: {self.config['name']}")
        else:
            print("⚠️ Configuración no encontrada, usando configuración por defecto")
            self.config = {
                'model_type': 'ultra_fast',
                'use_channels': 'all',
                'target_size': [48, 48, 24]
            }
        
        # Buscar modelos entrenados
        model_files = list(self.model_dir.glob('best_model_fold_*.pth'))
        
        print(f"🔄 Cargando {len(model_files)} modelos...")
        
        for i, model_path in enumerate(model_files):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Recrear modelo usando clases incluidas
                in_channels = {'pre_only': 1, 'post_only': 1, 'pre_post': 2, 'all': 3}[self.config['use_channels']]
                
                if self.config['model_type'] == 'fast_resnet_tiny':
                    model = FastResNet3D(in_channels=in_channels, model_size='tiny').to(device)
                elif self.config['model_type'] == 'fast_resnet_small':
                    model = FastResNet3D(in_channels=in_channels, model_size='small').to(device)
                else:  # ultra_fast o cualquier otro
                    model = UltraFastCNN3D(in_channels=in_channels).to(device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models.append(model)
                val_auc = checkpoint.get('val_auc', 'N/A')
                print(f"   ✅ Modelo {i+1}: Fold AUC {val_auc}")
                
            except Exception as e:
                print(f"   ❌ Error cargando modelo {i+1}: {e}")
        
        print(f"🎯 {len(self.models)} modelos cargados exitosamente")
    
    def predict_batch(self, batch: Dict) -> np.ndarray:
        """Hacer predicciones en un batch"""
        tensors = batch['tensor'].to(device, non_blocking=True)
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = torch.softmax(model(tensors), dim=1)
                predictions.append(outputs[:, 1])  # Probabilidad PCR positivo
        
        if predictions:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
            return ensemble_pred.cpu().numpy()
        else:
            return np.zeros(len(batch['patient_id']))
    
    def evaluate_dataset(self, dataset: CorrectedTestDataset, batch_size: int = 4) -> Dict:
        """Evaluar dataset completo"""
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        all_patient_ids = []
        all_predictions = []
        
        print(f"🔮 Evaluando con máscaras predichas...")
        print(f"📊 Dataset: {len(dataset)} pacientes")
        print(f"🧠 Ensemble: {len(self.models)} modelos")
        
        for batch_idx, batch in enumerate(dataloader):
            batch_predictions = self.predict_batch(batch)
            
            all_patient_ids.extend(batch['patient_id'])
            all_predictions.extend(batch_predictions)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"    Progreso: {(batch_idx + 1) * batch_size}/{len(dataset)}")
        
        return {
            'patient_ids': all_patient_ids,
            'predictions': all_predictions,
            'mask_type': 'predicted_segmentation',
            'model_config': self.config
        }

# =============================================================================
# FUNCIÓN PRINCIPAL CORREGIDA
# =============================================================================

def run_corrected_evaluation():
    """Evaluación corregida usando datos originales + predicciones"""
    
    print("🚀 EVALUACIÓN CORREGIDA CON PREDICCIONES")
    print("=" * 60)
    print("📁 Datos originales: C:/Users/usuario/Documents/Mama_Mia/datos/images")
    print("🔮 Predicciones: C:/Users/usuario/Documents/Mama_Mia/replicacion_definitiva/results_output")
    print("🎯 Solo PCR 0/1 (sin otros datos clínicos)")
    print("=" * 60)
    
    # PATHS CORREGIDOS
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    predictions_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output")
    model_dir = Path("D:/mama_mia_CORRECTED_HYBRID_results")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    pcr_labels_file = Path("D:/clinical_data_complete.json")
    
    # Verificaciones
    missing_paths = []
    for name, path in [
        ("Imágenes", images_dir),
        ("Predicciones", predictions_dir),
        ("Modelos", model_dir),
        ("Splits", splits_csv),
        ("PCR Labels", pcr_labels_file)
    ]:
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path}")
            missing_paths.append(name)
    
    if missing_paths:
        print(f"\n❌ Faltan: {missing_paths}")
        return None
    
    # Cargar pacientes de test
    df = pd.read_csv(splits_csv)
    test_patients = df['test_split'].dropna().astype(str).tolist()
    print(f"\n📊 Pacientes de test: {len(test_patients)}")
    
    # Crear dataset
    test_dataset = CorrectedTestDataset(
        images_dir=images_dir,
        predictions_dir=predictions_dir,
        patient_ids=test_patients,
        target_size=(48, 48, 24),
        use_channels='all'
    )
    
    if len(test_dataset) == 0:
        print("❌ Dataset vacío - revisa los paths")
        return None
    
    # Crear evaluador
    evaluator = CorrectedModelEvaluator(model_dir)
    
    if len(evaluator.models) == 0:
        print("❌ No se cargaron modelos")
        return None
    
    # Hacer predicciones
    prediction_results = evaluator.evaluate_dataset(test_dataset)
    
    # Cargar SOLO PCR labels (0 o 1)
    print(f"\n📋 Cargando PCR labels (SOLO 0 o 1)...")
    with open(pcr_labels_file, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {}
    for item in pcr_list:
        if 'patient_id' in item and 'pcr' in item:
            pcr_value = item['pcr']
            if pcr_value in ["0", "1"]:  # SOLO PCR 0 o 1
                pcr_data[item['patient_id']] = int(pcr_value)
    
    print(f"✅ PCR labels válidos: {len(pcr_data)} pacientes")
    
    # Alinear predicciones con PCR ground truth
    valid_predictions = []
    pred_scores = []
    true_labels = []
    
    for pid, pred in zip(prediction_results['patient_ids'], prediction_results['predictions']):
        if pid in pcr_data:
            valid_predictions.append({
                'patient_id': pid,
                'predicted_prob': float(pred),
                'predicted_class': int(pred > 0.5),
                'true_pcr': pcr_data[pid],
                'correct': int(pred > 0.5) == pcr_data[pid]
            })
            pred_scores.append(float(pred))
            true_labels.append(pcr_data[pid])
    
    if not valid_predictions:
        print("❌ No hay predicciones válidas para evaluar")
        return None
    
    # Calcular métricas
    auc = roc_auc_score(true_labels, pred_scores)
    accuracy = accuracy_score(true_labels, [p['predicted_class'] for p in valid_predictions])
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   🎯 AUC: {auc:.4f}")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   📈 Pacientes evaluados: {len(valid_predictions)}")
    print(f"   🔮 Método: Máscaras predichas (realista)")
    
    # Comparaciones
    baselines = {
        "Random": 0.5000,
        "Tu baseline anterior": 0.5763,
        "Objetivo": 0.6500
    }
    
    print(f"\n📈 COMPARACIONES:")
    for name, baseline in baselines.items():
        diff = auc - baseline
        if diff > 0:
            print(f"   ✅ vs {name}: {baseline:.4f} → {auc:.4f} (+{diff:.4f})")
        else:
            print(f"   📉 vs {name}: {baseline:.4f} → {auc:.4f} ({diff:.4f})")
    
    # Guardar resultados
    results = {
        'evaluation_method': 'predicted_masks_with_original_images',
        'metrics': {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'total_patients': len(valid_predictions)
        },
        'data_sources': {
            'images': str(images_dir),
            'predictions': str(predictions_dir),
            'models': str(model_dir)
        },
        'predictions': valid_predictions,
        'model_info': {
            'config': prediction_results['model_config'],
            'ensemble_size': len(evaluator.models)
        }
    }
    
    output_file = Path("D:/corrected_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV
    csv_file = output_file.with_suffix('.csv')
    df_results = pd.DataFrame(valid_predictions)
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n💾 Resultados guardados:")
    print(f"   📄 JSON: {output_file}")
    print(f"   📊 CSV: {csv_file}")
    
    print(f"\n✅ EVALUACIÓN COMPLETADA EXITOSAMENTE!")
    
    return results

if __name__ == "__main__":
    run_corrected_evaluation()