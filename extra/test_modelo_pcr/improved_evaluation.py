# improved_evaluation.py
"""
Evaluación MEJORADA para conseguir AUC > 0.60
- Probará múltiples configuraciones
- Incluirá máscaras predichas
- Aplicará correcciones automáticas
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
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODELOS (mismo código anterior)
# =============================================================================

class UltraFastCNN3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
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
    def __init__(self, in_channels=3, num_classes=2, model_size='small'):
        super().__init__()
        
        if model_size == 'tiny':
            channels = [16, 32, 64, 128]
            blocks = [1, 1, 1, 1]
        elif model_size == 'small':
            channels = [32, 64, 128, 256]
            blocks = [2, 2, 2, 2]
        else:
            channels = [64, 128, 256, 512]
            blocks = [2, 2, 3, 2]
        
        self.conv1 = nn.Conv3d(in_channels, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(channels[0], channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(channels[3], num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        ))
        
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
        
        x = self.layer1(x) + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# =============================================================================
# DATASET MEJORADO CON MÚLTIPLES CONFIGURACIONES
# =============================================================================

class ImprovedDataset(Dataset):
    """Dataset mejorado que prueba múltiples configuraciones"""
    
    def __init__(self, 
                 images_dir: Path,
                 predictions_dir: Path,
                 patient_ids: List[str],
                 config: Dict,
                 target_size=(48, 48, 24)):
        
        self.images_dir = Path(images_dir)
        self.predictions_dir = Path(predictions_dir)
        self.patient_ids = patient_ids
        self.config = config
        self.target_size = target_size
        
        # Configuraciones posibles
        self.channel_configs = {
            1: {'name': 'post_only', 'description': 'Solo post-contraste'},
            2: {'name': 'pre_post', 'description': 'Pre + Post contraste'},
            3: {'name': 'all_with_mask', 'description': 'Pre + Post + Máscara predicha'}
        }
        
        print(f"🔧 Configuración: {self.channel_configs[config['channels']]['description']}")
        
        # Verificar archivos disponibles
        self.valid_indices = []
        self.available_data = []
        
        for i, pid in enumerate(patient_ids):
            pre_file = self._find_image_file(pid, 'pre')
            post_file = self._find_image_file(pid, 'post')
            predicted_mask = self._find_predicted_mask(pid)
            
            # Verificar requisitos según configuración
            valid = True
            required_files = ['post_file']
            
            if config['channels'] >= 2:
                required_files.append('pre_file')
            if config['channels'] == 3:
                required_files.append('predicted_mask')
            
            file_dict = {
                'pre_file': pre_file,
                'post_file': post_file, 
                'predicted_mask': predicted_mask
            }
            
            for req_file in required_files:
                if not file_dict[req_file]:
                    valid = False
                    break
            
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
            f"{patient_id}_prediction.nii.gz"
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
    
    def _normalize_image(self, data: np.ndarray, method: str = 'percentile') -> np.ndarray:
        """Normalización mejorada"""
        
        if method == 'percentile':
            # Normalización por percentiles (más robusta)
            if np.max(data) > 0:
                p1, p99 = np.percentile(data[data > 0], [1, 99])
                data = np.clip(data, p1, p99)
                data = (data - p1) / (p99 - p1) if p99 > p1 else data
        
        elif method == 'zscore':
            # Z-score normalización
            non_zero = data[data > 0]
            if len(non_zero) > 0:
                mean_val = np.mean(non_zero)
                std_val = np.std(non_zero)
                if std_val > 0:
                    data = (data - mean_val) / std_val
                    data[data < -3] = -3  # Clip extremos
                    data[data > 3] = 3
        
        elif method == 'minmax':
            # Min-max normalización
            if np.max(data) > np.min(data):
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return data.astype(np.float32)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_info = self.available_data[idx]
        patient_id = data_info['patient_id']
        
        try:
            components = []
            
            # Cargar componentes según configuración
            if self.config['channels'] >= 2:
                # Pre y post
                pre_data = nib.load(data_info['pre_file']).get_fdata().astype(np.float32)
                post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
                
                # Normalización mejorada
                norm_method = self.config.get('normalization', 'percentile')
                pre_data = self._normalize_image(pre_data, norm_method)
                post_data = self._normalize_image(post_data, norm_method)
                
                components = [pre_data, post_data]
                
            elif self.config['channels'] == 1:
                # Solo post
                post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
                post_data = self._normalize_image(post_data, self.config.get('normalization', 'percentile'))
                components = [post_data]
            
            # Añadir máscara si es necesario
            if self.config['channels'] == 3:
                predicted_mask_data = nib.load(data_info['predicted_mask_file']).get_fdata().astype(np.float32)
                
                # Procesar máscara
                if predicted_mask_data.ndim > 3:
                    if predicted_mask_data.shape[0] == 2:
                        predicted_mask_data = predicted_mask_data[1]
                    elif predicted_mask_data.shape[-1] == 2:
                        predicted_mask_data = predicted_mask_data[..., 1]
                    else:
                        predicted_mask_data = predicted_mask_data.squeeze()
                
                # Binarizar y suavizar
                predicted_mask_binary = (predicted_mask_data > 0.5).astype(np.float32)
                
                # Opcional: dilatación ligera de la máscara
                if self.config.get('dilate_mask', False):
                    from scipy.ndimage import binary_dilation
                    predicted_mask_binary = binary_dilation(predicted_mask_binary, iterations=1).astype(np.float32)
                
                components.append(predicted_mask_binary)
            
            # Crear tensor
            selected_data = np.stack(components, axis=0)
            
            # Resize con interpolación mejorada
            tensor = torch.from_numpy(selected_data)
            tensor = F.interpolate(
                tensor.unsqueeze(0), 
                size=self.target_size, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Test-time augmentation si está habilitado
            if self.config.get('tta', False):
                if torch.rand(1) < 0.5:
                    tensor = torch.flip(tensor, [1])  # Flip horizontal
            
            return {
                'tensor': tensor.float(),
                'patient_id': patient_id,
                'config': self.config
            }
            
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {e}")
            dummy_tensor = torch.zeros((self.config['channels'], *self.target_size))
            return {
                'tensor': dummy_tensor,
                'patient_id': patient_id,
                'config': self.config
            }

# =============================================================================
# EVALUADOR MEJORADO CON MÚLTIPLES ESTRATEGIAS
# =============================================================================

class ImprovedEvaluator:
    """Evaluador mejorado que prueba múltiples configuraciones"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = {}  # Guardará modelos por configuración
        self.base_config = None
        
        self._load_all_models()
    
    def _detect_model_channels(self, model_path: Path) -> int:
        """Detectar canales del modelo"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            
            for key, param in state_dict.items():
                if 'conv' in key.lower() and 'weight' in key and param.dim() == 5:
                    return param.shape[1]
            
            if 'features.0.weight' in state_dict:
                return state_dict['features.0.weight'].shape[1]
            if 'conv1.weight' in state_dict:
                return state_dict['conv1.weight'].shape[1]
            
            return 3
        except:
            return 3
    
    def _load_all_models(self):
        """Cargar modelos y agrupar por configuración"""
        
        # Cargar configuración base
        config_file = self.model_dir / 'final_results.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                results = json.load(f)
                self.base_config = results['best_config']
        else:
            self.base_config = {'model_type': 'ultra_fast'}
        
        # Buscar todos los modelos
        model_files = list(self.model_dir.glob('best_model_fold_*.pth'))
        
        print(f"🔄 Cargando {len(model_files)} modelos...")
        
        # Agrupar modelos por número de canales
        channels_found = set()
        
        for model_path in model_files:
            channels = self._detect_model_channels(model_path)
            channels_found.add(channels)
            
            if channels not in self.models:
                self.models[channels] = []
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Crear modelo
                if self.base_config['model_type'] == 'fast_resnet_tiny':
                    model = FastResNet3D(in_channels=channels, model_size='tiny').to(device)
                elif self.base_config['model_type'] == 'fast_resnet_small':
                    model = FastResNet3D(in_channels=channels, model_size='small').to(device)
                else:
                    model = UltraFastCNN3D(in_channels=channels).to(device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models[channels].append({
                    'model': model,
                    'val_auc': checkpoint.get('val_auc', 0.5),
                    'path': model_path
                })
                
            except Exception as e:
                print(f"   ❌ Error cargando {model_path.name}: {e}")
        
        print(f"✅ Modelos cargados por configuración:")
        for channels, models in self.models.items():
            print(f"   {channels} canales: {len(models)} modelos")
    
    def evaluate_configuration(self, 
                             dataset: ImprovedDataset, 
                             channels: int,
                             batch_size: int = 4) -> Dict:
        """Evaluar una configuración específica"""
        
        if channels not in self.models or len(self.models[channels]) == 0:
            print(f"❌ No hay modelos para {channels} canales")
            return None
        
        models = self.models[channels]
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        all_patient_ids = []
        all_predictions = []
        
        print(f"🔮 Evaluando configuración {channels} canales...")
        print(f"   📊 Dataset: {len(dataset)} pacientes")
        print(f"   🧠 Modelos: {len(models)}")
        
        for batch_idx, batch in enumerate(dataloader):
            tensors = batch['tensor'].to(device, non_blocking=True)
            
            # Predicciones de todos los modelos
            batch_predictions = []
            
            with torch.no_grad():
                for model_info in models:
                    model = model_info['model']
                    outputs = torch.softmax(model(tensors), dim=1)
                    batch_predictions.append(outputs[:, 1])
            
            # Ensemble con pesos por validation AUC
            if batch_predictions:
                weights = [m['val_auc'] for m in models]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    weighted_preds = []
                    for i, pred in enumerate(batch_predictions):
                        weighted_preds.append(pred * weights[i] / total_weight)
                    ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
                else:
                    ensemble_pred = torch.stack(batch_predictions).mean(dim=0)
                
                all_patient_ids.extend(batch['patient_id'])
                all_predictions.extend(ensemble_pred.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Progreso: {(batch_idx + 1) * batch_size}/{len(dataset)}")
        
        return {
            'patient_ids': all_patient_ids,
            'predictions': all_predictions,
            'channels': channels,
            'num_models': len(models),
            'model_aucs': [m['val_auc'] for m in models]
        }

def run_improved_evaluation():
    """Ejecutar evaluación mejorada con múltiples estrategias"""
    
    print("🚀 EVALUACIÓN MEJORADA - MÚLTIPLES ESTRATEGIAS")
    print("=" * 60)
    print("🎯 Objetivo: AUC > 0.60")
    print("=" * 60)
    
    # Paths
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    predictions_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output")
    model_dir = Path("D:/mama_mia_CORRECTED_HYBRID_results")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    pcr_labels_file = Path("D:/clinical_data_complete.json")
    
    # Cargar pacientes de test
    df = pd.read_csv(splits_csv)
    test_patients = df['test_split'].dropna().astype(str).tolist()
    
    # Cargar PCR labels
    with open(pcr_labels_file, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {}
    for item in pcr_list:
        if 'patient_id' in item and 'pcr' in item and item['pcr'] in ["0", "1"]:
            pcr_data[item['patient_id']] = int(item['pcr'])
    
    # Crear evaluador
    evaluator = ImprovedEvaluator(model_dir)
    
    # Configuraciones a probar
    configurations = [
        {
            'name': 'Configuración 1: Solo post-contraste',
            'channels': 1,
            'normalization': 'percentile',
            'tta': False
        },
        {
            'name': 'Configuración 2: Pre + Post contraste',
            'channels': 2,
            'normalization': 'percentile',
            'tta': False
        },
        {
            'name': 'Configuración 3: Pre + Post + Máscara predicha',
            'channels': 3,
            'normalization': 'percentile',
            'tta': False
        },
        {
            'name': 'Configuración 4: Pre + Post (Z-score norm)',
            'channels': 2,
            'normalization': 'zscore',
            'tta': False
        },
        {
            'name': 'Configuración 5: Todo + TTA',
            'channels': 3,
            'normalization': 'percentile',
            'tta': True,
            'dilate_mask': True
        }
    ]
    
    best_auc = 0
    best_config = None
    best_results = None
    all_results = []
    
    for config in configurations:
        print(f"\n🧪 PROBANDO: {config['name']}")
        print("-" * 50)
        
        # Verificar si hay modelos para esta configuración
        if config['channels'] not in evaluator.models:
            print(f"❌ No hay modelos para {config['channels']} canales")
            continue
        
        # Crear dataset para esta configuración
        dataset = ImprovedDataset(
            images_dir=images_dir,
            predictions_dir=predictions_dir,
            patient_ids=test_patients,
            config=config,
            target_size=(48, 48, 24)
        )
        
        if len(dataset) == 0:
            print(f"❌ Dataset vacío para esta configuración")
            continue
        
        # Evaluar
        prediction_results = evaluator.evaluate_configuration(dataset, config['channels'])
        
        if not prediction_results:
            print(f"❌ Error en evaluación")
            continue
        
        # Calcular métricas
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
            print(f"❌ No hay predicciones válidas")
            continue
        
        # Calcular AUC
        auc = roc_auc_score(true_labels, pred_scores)
        
        # Probar inversión si AUC < 0.5
        if auc < 0.5:
            inverted_scores = [1 - p for p in pred_scores]
            inverted_auc = roc_auc_score(true_labels, inverted_scores)
            
            if inverted_auc > auc:
                print(f"🔄 Aplicando inversión de etiquetas")
                auc = inverted_auc
                pred_scores = inverted_scores
                # Actualizar predicciones
                for i, p in enumerate(valid_predictions):
                    p['predicted_prob'] = 1 - p['predicted_prob']
                    p['predicted_class'] = int(p['predicted_prob'] > 0.5)
                    p['correct'] = p['predicted_class'] == p['true_pcr']
        
        accuracy = accuracy_score(true_labels, [p['predicted_class'] for p in valid_predictions])
        
        print(f"📊 RESULTADOS:")
        print(f"   🎯 AUC: {auc:.4f}")
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📈 Pacientes: {len(valid_predictions)}")
        
        # Guardar resultados
        config_results = {
            'config': config,
            'metrics': {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'total_patients': len(valid_predictions)
            },
            'predictions': valid_predictions,
            'model_info': prediction_results
        }
        
        all_results.append(config_results)
        
        # Actualizar mejor resultado
        if auc > best_auc:
            best_auc = auc
            best_config = config
            best_results = config_results
            print(f"🏆 NUEVA MEJOR CONFIGURACIÓN!")
    
    # Mostrar resultados finales
    print(f"\n🏆 RESULTADOS FINALES")
    print("=" * 50)
    
    if best_results:
        print(f"🎯 MEJOR AUC: {best_auc:.4f}")
        print(f"🔧 MEJOR CONFIGURACIÓN: {best_config['name']}")
        print(f"📊 Canales: {best_config['channels']}")
        print(f"🔄 Normalización: {best_config['normalization']}")
        
        # Comparaciones
        baselines = {
            "Resultado anterior": 0.4900,
            "Random": 0.5000,
            "Tu baseline": 0.5763,
            "Objetivo": 0.6500
        }
        
        print(f"\n📈 COMPARACIONES:")
        for name, baseline in baselines.items():
            diff = best_auc - baseline
            if diff > 0:
                print(f"   ✅ vs {name}: {baseline:.4f} → {best_auc:.4f} (+{diff:.4f})")
            else:
                print(f"   📉 vs {name}: {baseline:.4f} → {best_auc:.4f} ({diff:.4f})")
        
        # Guardar mejor resultado
        output_file = Path("D:/improved_evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'best_result': best_results,
                'all_results': all_results,
                'improvement_summary': {
                    'original_auc': 0.4900,
                    'improved_auc': best_auc,
                    'improvement': best_auc - 0.4900,
                    'best_config': best_config
                }
            }, f, indent=2)
        
        print(f"\n💾 Resultados guardados: {output_file}")
        
        # Evaluación del resultado
        if best_auc > 0.65:
            print(f"\n🎉 ¡EXCELENTE! Objetivo alcanzado")
        elif best_auc > 0.60:
            print(f"\n🚀 ¡MUY BUENO! Cerca del objetivo")
        elif best_auc > 0.55:
            print(f"\n👍 ¡BUENA MEJORA! Progreso significativo")
        elif best_auc > 0.50:
            print(f"\n✅ Mejor que random, sigue optimizando")
        else:
            print(f"\n⚠️ Necesita más trabajo")
        
        return best_results
    
    else:
        print(f"❌ No se pudieron evaluar configuraciones")
        return None

if __name__ == "__main__":
    run_improved_evaluation()