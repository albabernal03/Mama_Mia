# fixed_channel_evaluation.py
"""
Evaluación CORREGIDA - Detecta automáticamente el número de canales del modelo
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
# MODELOS INCLUIDOS DIRECTAMENTE
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
# FUNCIÓN PARA DETECTAR CANALES DEL MODELO
# =============================================================================

def detect_model_channels(model_path: Path) -> int:
    """Detectar automáticamente cuántos canales espera el modelo"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Buscar la primera capa convolucional
        for key, param in state_dict.items():
            if 'conv' in key.lower() and 'weight' in key and param.dim() == 5:
                # Para conv3d: [out_channels, in_channels, depth, height, width]
                in_channels = param.shape[1]
                print(f"🔍 Detectado: {in_channels} canales de entrada en {key}")
                return in_channels
        
        # Si no encuentra, buscar en features.0.weight (UltraFastCNN3D)
        if 'features.0.weight' in state_dict:
            in_channels = state_dict['features.0.weight'].shape[1]
            print(f"🔍 Detectado: {in_channels} canales en features.0.weight")
            return in_channels
        
        # Si no encuentra, buscar en conv1.weight (FastResNet3D)
        if 'conv1.weight' in state_dict:
            in_channels = state_dict['conv1.weight'].shape[1]
            print(f"🔍 Detectado: {in_channels} canales en conv1.weight")
            return in_channels
        
        print("⚠️ No se pudo detectar canales, usando 3 por defecto")
        return 3
        
    except Exception as e:
        print(f"❌ Error detectando canales: {e}")
        return 3

# =============================================================================
# DATASET CON DETECCIÓN AUTOMÁTICA DE CANALES
# =============================================================================

class SmartChannelDataset(Dataset):
    """Dataset que se adapta automáticamente al número de canales del modelo"""
    
    def __init__(self, 
                 images_dir: Path,
                 predictions_dir: Path,
                 patient_ids: List[str],
                 required_channels: int,  # ← NUEVO: número de canales requerido
                 target_size=(48, 48, 24)):
        
        self.images_dir = Path(images_dir)
        self.predictions_dir = Path(predictions_dir)
        self.patient_ids = patient_ids
        self.required_channels = required_channels
        self.target_size = target_size
        
        # Mapear número de canales a configuración
        self.channel_mapping = {
            1: 'post_only',
            2: 'pre_post', 
            3: 'all'
        }
        
        self.use_channels = self.channel_mapping.get(required_channels, 'all')
        
        print(f"🔧 Configuración automática:")
        print(f"   🎯 Canales requeridos: {required_channels}")
        print(f"   📺 Modo de canales: {self.use_channels}")
        
        # Verificar archivos disponibles
        self.valid_indices = []
        self.available_data = []
        
        for i, pid in enumerate(patient_ids):
            pre_file = self._find_image_file(pid, 'pre')
            post_file = self._find_image_file(pid, 'post')
            predicted_mask = self._find_predicted_mask(pid)
            
            # Verificar qué archivos necesitamos según el número de canales
            valid = True
            if required_channels >= 2 and not pre_file:
                valid = False
            if not post_file:
                valid = False
            if required_channels == 3 and not predicted_mask:
                valid = False
                
            if valid:
                self.valid_indices.append(i)
                self.available_data.append({
                    'patient_id': pid,
                    'pre_file': pre_file,
                    'post_file': post_file,
                    'predicted_mask_file': predicted_mask
                })
        
        print(f"📊 Dataset válido: {len(self.valid_indices)} pacientes")
    
    def _find_image_file(self, patient_id: str, phase: str) -> Optional[Path]:
        """Buscar archivo de imagen"""
        phase_patterns = {
            'pre': ['_0000.nii.gz', '_pre.nii.gz', '_T1.nii.gz', 'pre.nii.gz'],
            'post': ['_0001.nii.gz', '_post.nii.gz', '_T1c.nii.gz', 'post.nii.gz', '.nii.gz']
        }
        
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
                    file_path = search_dir / f"{patient_id}{pattern}"
                    if file_path.exists():
                        return file_path
                
                for nii_file in search_dir.glob("*.nii.gz"):
                    filename = nii_file.name.lower()
                    if patient_id.lower() in filename:
                        for pattern in phase_patterns[phase]:
                            if pattern.replace('.nii.gz', '') in filename:
                                return nii_file
        return None
    
    def _find_predicted_mask(self, patient_id: str) -> Optional[Path]:
        """Buscar predicción de segmentación"""
        possible_patterns = [
            f"{patient_id}.nii.gz",
            f"{patient_id}_pred.nii.gz",
            f"{patient_id}_prediction.nii.gz", 
            f"{patient_id}_seg.nii.gz"
        ]
        
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
                
                for nii_file in search_dir.glob("**/*.nii.gz"):
                    if patient_id in nii_file.name:
                        return nii_file
        return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_info = self.available_data[idx]
        patient_id = data_info['patient_id']
        
        try:
            # Cargar datos necesarios según número de canales
            components = []
            
            if self.required_channels >= 2:
                # Necesitamos pre y post
                pre_data = nib.load(data_info['pre_file']).get_fdata().astype(np.float32)
                post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
                
                # Normalización simple
                if np.max(pre_data) > 0:
                    pre_data = pre_data / np.percentile(pre_data[pre_data > 0], 99)
                if np.max(post_data) > 0:
                    post_data = post_data / np.percentile(post_data[post_data > 0], 99)
                
                components = [pre_data, post_data]
                
            elif self.required_channels == 1:
                # Solo post-contraste
                post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
                if np.max(post_data) > 0:
                    post_data = post_data / np.percentile(post_data[post_data > 0], 99)
                components = [post_data]
            
            # Si necesitamos 3 canales, añadir máscara predicha
            if self.required_channels == 3:
                predicted_mask_data = nib.load(data_info['predicted_mask_file']).get_fdata().astype(np.float32)
                
                # Procesar máscara predicha
                if predicted_mask_data.ndim > 3:
                    if predicted_mask_data.shape[0] == 2:
                        predicted_mask_data = predicted_mask_data[1]
                    elif predicted_mask_data.shape[-1] == 2:
                        predicted_mask_data = predicted_mask_data[..., 1]
                    else:
                        predicted_mask_data = predicted_mask_data[0] if predicted_mask_data.shape[0] > 1 else predicted_mask_data.squeeze()
                
                predicted_mask_binary = (predicted_mask_data > 0.5).astype(np.float32)
                components.append(predicted_mask_binary)
            
            # Crear tensor con el número exacto de canales requerido
            selected_data = np.stack(components, axis=0)
            
            # Resize
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
                'channels_used': self.required_channels
            }
            
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {e}")
            dummy_tensor = torch.zeros((self.required_channels, *self.target_size))
            return {
                'tensor': dummy_tensor,
                'patient_id': patient_id,
                'channels_used': self.required_channels
            }

# =============================================================================
# EVALUADOR CORREGIDO CON DETECCIÓN AUTOMÁTICA
# =============================================================================

class SmartModelEvaluator:
    """Evaluador que detecta automáticamente la configuración del modelo"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = []
        self.config = None
        self.detected_channels = None
        
        self._load_models()
    
    def _load_models(self):
        """Cargar modelos detectando automáticamente los canales"""
        
        # Cargar configuración
        config_file = self.model_dir / 'final_results.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                results = json.load(f)
                self.config = results['best_config']
                print(f"✅ Configuración cargada: {self.config['name']}")
        else:
            print("⚠️ Configuración no encontrada")
            self.config = {'model_type': 'ultra_fast'}
        
        # Buscar modelos entrenados
        model_files = list(self.model_dir.glob('best_model_fold_*.pth'))
        
        if not model_files:
            print("❌ No se encontraron modelos")
            return
        
        # Detectar canales del primer modelo
        self.detected_channels = detect_model_channels(model_files[0])
        print(f"🔧 Canales detectados: {self.detected_channels}")
        
        print(f"🔄 Cargando {len(model_files)} modelos...")
        
        for i, model_path in enumerate(model_files):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Crear modelo con el número correcto de canales
                if self.config['model_type'] == 'fast_resnet_tiny':
                    model = FastResNet3D(in_channels=self.detected_channels, model_size='tiny').to(device)
                elif self.config['model_type'] == 'fast_resnet_small':
                    model = FastResNet3D(in_channels=self.detected_channels, model_size='small').to(device)
                else:
                    model = UltraFastCNN3D(in_channels=self.detected_channels).to(device)
                
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
                predictions.append(outputs[:, 1])
        
        if predictions:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
            return ensemble_pred.cpu().numpy()
        else:
            return np.zeros(len(batch['patient_id']))
    
    def evaluate_dataset(self, dataset: SmartChannelDataset, batch_size: int = 4) -> Dict:
        """Evaluar dataset completo"""
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        all_patient_ids = []
        all_predictions = []
        
        print(f"🔮 Evaluando con configuración automática...")
        print(f"📊 Dataset: {len(dataset)} pacientes")
        print(f"🧠 Ensemble: {len(self.models)} modelos")
        print(f"🎯 Canales: {self.detected_channels}")
        
        for batch_idx, batch in enumerate(dataloader):
            batch_predictions = self.predict_batch(batch)
            
            all_patient_ids.extend(batch['patient_id'])
            all_predictions.extend(batch_predictions)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"    Progreso: {(batch_idx + 1) * batch_size}/{len(dataset)}")
        
        return {
            'patient_ids': all_patient_ids,
            'predictions': all_predictions,
            'detected_channels': self.detected_channels,
            'model_config': self.config
        }

# =============================================================================
# FUNCIÓN PRINCIPAL CORREGIDA
# =============================================================================

def run_smart_evaluation():
    """Evaluación inteligente que detecta automáticamente la configuración"""
    
    print("🧠 EVALUACIÓN INTELIGENTE - DETECCIÓN AUTOMÁTICA")
    print("=" * 60)
    print("🔧 Detecta automáticamente canales del modelo")
    print("📊 Se adapta a la configuración real")
    print("=" * 60)
    
    # PATHS
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    predictions_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output")
    model_dir = Path("D:/mama_mia_CORRECTED_HYBRID_results")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    pcr_labels_file = Path("D:/clinical_data_complete.json")
    
    # Verificaciones básicas
    missing_paths = []
    for name, path in [
        ("Imágenes", images_dir),
        ("Predicciones", predictions_dir),
        ("Modelos", model_dir),
        ("Splits", splits_csv),
        ("PCR Labels", pcr_labels_file)
    ]:
        if path.exists():
            print(f"✅ {name}: OK")
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
    
    # Crear evaluador (detecta canales automáticamente)
    evaluator = SmartModelEvaluator(model_dir)
    
    if len(evaluator.models) == 0:
        print("❌ No se cargaron modelos")
        return None
    
    # Crear dataset adaptativo
    test_dataset = SmartChannelDataset(
        images_dir=images_dir,
        predictions_dir=predictions_dir,
        patient_ids=test_patients,
        required_channels=evaluator.detected_channels,  # ← ADAPTATIVO
        target_size=(48, 48, 24)
    )
    
    if len(test_dataset) == 0:
        print("❌ Dataset vacío")
        return None
    
    # Hacer predicciones
    prediction_results = evaluator.evaluate_dataset(test_dataset)
    
    # Cargar PCR labels
    print(f"\n📋 Cargando PCR labels...")
    with open(pcr_labels_file, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {}
    for item in pcr_list:
        if 'patient_id' in item and 'pcr' in item:
            pcr_value = item['pcr']
            if pcr_value in ["0", "1"]:
                pcr_data[item['patient_id']] = int(pcr_value)
    
    print(f"✅ PCR labels válidos: {len(pcr_data)} pacientes")
    
    # Alinear predicciones con PCR
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
        print("❌ No hay predicciones válidas")
        return None
    
    # Calcular métricas
    auc = roc_auc_score(true_labels, pred_scores)
    accuracy = accuracy_score(true_labels, [p['predicted_class'] for p in valid_predictions])
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   🎯 AUC: {auc:.4f}")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   📈 Pacientes evaluados: {len(valid_predictions)}")
    print(f"   🔧 Canales detectados: {prediction_results['detected_channels']}")
    
    # Identificar configuración real
    channel_names = {1: 'post_only', 2: 'pre_post', 3: 'all (pre+post+mask)'}
    detected_config = channel_names.get(prediction_results['detected_channels'], 'unknown')
    print(f"   📺 Configuración real: {detected_config}")
    
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
        'evaluation_method': 'smart_channel_detection',
        'detected_configuration': {
            'channels': prediction_results['detected_channels'],
            'channel_mode': detected_config
        },
        'metrics': {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'total_patients': len(valid_predictions)
        },
        'predictions': valid_predictions
    }
    
    output_file = Path("D:/smart_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    csv_file = output_file.with_suffix('.csv')
    df_results = pd.DataFrame(valid_predictions)
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n💾 Resultados guardados:")
    print(f"   📄 JSON: {output_file}")
    print(f"   📊 CSV: {csv_file}")
    
    print(f"\n✅ EVALUACIÓN INTELIGENTE COMPLETADA!")
    
    return results

if __name__ == "__main__":
    run_smart_evaluation()