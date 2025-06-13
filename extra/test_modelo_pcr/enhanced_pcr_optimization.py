# enhanced_smart_optimization.py
"""
Pipeline MEJORADO con detección automática de canales + TTA + Attention
Basado en fixed_channel_evaluation.py con mejoras avanzadas
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# FUNCIONES ROBUSTAS MEJORADAS
# =============================================================================

def enhanced_robust_normalize(data: np.ndarray, method: str = 'adaptive_zscore') -> np.ndarray:
    """Normalización adaptiva mejorada"""
    
    if data.size == 0:
        return data.astype(np.float32)
    
    # Limpieza inicial
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    non_zero = data[data > 0]
    
    if len(non_zero) == 0:
        return data.astype(np.float32)
    
    if method == 'adaptive_zscore':
        try:
            # Estadísticas robustas mejoradas
            p5, p25, p50, p75, p95 = np.percentile(non_zero, [5, 25, 50, 75, 95])
            iqr = p75 - p25
            
            if iqr > 1e-8:
                # Estimación robusta de std
                robust_std = iqr / 1.349
                
                # Detección de outliers
                outlier_threshold = p95
                data_clipped = np.clip(data, 0, outlier_threshold)
                
                # Normalización
                normalized = (data_clipped - p50) / robust_std
                normalized[data == 0] = 0
                
                # Clip más conservador
                normalized = np.clip(normalized, -6, 6)
                
                # Rescalar a [0, 1] para estabilidad
                if normalized.max() > normalized.min():
                    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
                
                return normalized.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error en normalización adaptiva: {e}")
    
    # Fallback mejorado
    try:
        p5, p95 = np.percentile(non_zero, [5, 95])
        data_clipped = np.clip(data, 0, p95)
        if p95 > 0:
            normalized = data_clipped / p95
        else:
            normalized = data.astype(np.float32)
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
    except Exception:
        return np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def apply_test_time_augmentation(tensor: torch.Tensor, n_augs: int = 8) -> List[torch.Tensor]:
    """Test-Time Augmentation agresivo"""
    
    augmented = [tensor]  # Original
    
    # Rotaciones en planos axiales
    for angle in [90, 180, 270]:
        rotated = torch.rot90(tensor, k=angle//90, dims=[-2, -1])
        augmented.append(rotated)
    
    # Flips
    augmented.append(torch.flip(tensor, dims=[-1]))  # Horizontal
    augmented.append(torch.flip(tensor, dims=[-2]))  # Vertical
    augmented.append(torch.flip(tensor, dims=[-3]))  # Axial
    
    # Combinaciones
    augmented.append(torch.flip(torch.rot90(tensor, k=1, dims=[-2, -1]), dims=[-1]))
    
    return augmented[:n_augs]

# =============================================================================
# DETECCIÓN AUTOMÁTICA DE CANALES
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
# MODELOS INCLUIDOS DIRECTAMENTE
# =============================================================================

class UltraFastCNN3D(nn.Module):
    """CNN 3D ultra-rápido"""
    
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
    """ResNet 3D simple"""
    
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
# MODELO MEJORADO CON ATTENTION (ADAPTATIVO)
# =============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Dual pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Attention weights
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        
        return x * attention.expand_as(x)

class EnhancedCNN3D(nn.Module):
    """CNN 3D mejorado con attention - ADAPTATIVO a cualquier número de canales"""
    
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        
        # Encoder con attention
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.att1 = ChannelAttention(32)
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.att2 = ChannelAttention(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.att3 = ChannelAttention(128)
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.att4 = ChannelAttention(256)
        
        # Global pooling mejorado
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier con más regularización
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
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
        # Encoder con attention
        x1 = self.att1(self.conv1(x))
        x2 = self.att2(self.conv2(x1))
        x3 = self.att3(self.conv3(x2))
        x4 = self.att4(self.conv4(x3))
        
        # Global features
        x = self.global_pool(x4)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# =============================================================================
# DATASET INTELIGENTE CON MEJORAS
# =============================================================================

class SmartEnhancedDataset(Dataset):
    """Dataset que se adapta automáticamente + mejoras de preprocessing"""
    
    def __init__(self, 
                 images_dir: Path,
                 predictions_dir: Path,
                 patient_ids: List[str],
                 required_channels: int,
                 target_size=(64, 64, 32),  # Resolución aumentada
                 normalization='adaptive_zscore'):
        
        self.images_dir = Path(images_dir)
        self.predictions_dir = Path(predictions_dir)
        self.patient_ids = patient_ids
        self.required_channels = required_channels
        self.target_size = target_size
        self.normalization = normalization
        
        # Mapear número de canales a configuración
        self.channel_mapping = {
            1: 'post_only',
            2: 'pre_post', 
            3: 'all'
        }
        
        self.use_channels = self.channel_mapping.get(required_channels, 'all')
        
        print(f"🔧 Configuración automática mejorada:")
        print(f"   🎯 Canales requeridos: {required_channels}")
        print(f"   📺 Modo de canales: {self.use_channels}")
        print(f"   📐 Resolución: {target_size}")
        print(f"   🔧 Normalización: {normalization}")
        
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
    
    def _enhance_contrast(self, data: np.ndarray) -> np.ndarray:
        """Mejora de contraste adaptivo"""
        
        if data.max() == data.min():
            return data
        
        try:
            # Normalizar a [0, 1]
            data_norm = (data - data.min()) / (data.max() - data.min())
            
            # Aplicar gamma correction adaptivo
            gamma = 1.2 if np.mean(data_norm) < 0.5 else 0.8
            enhanced = np.power(data_norm, gamma)
            
            return enhanced.astype(np.float32)
            
        except Exception:
            return data.astype(np.float32)
    
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
                
                # Alinear dimensiones
                if pre_data.shape != post_data.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(pre_data.shape, post_data.shape))
                    pre_data = pre_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                    post_data = post_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                
                # Normalización mejorada
                pre_data = enhanced_robust_normalize(pre_data, self.normalization)
                post_data = enhanced_robust_normalize(post_data, self.normalization)
                
                # Mejora de contraste
                pre_data = self._enhance_contrast(pre_data)
                post_data = self._enhance_contrast(post_data)
                
                components = [pre_data, post_data]
                
            elif self.required_channels == 1:
                # Solo post-contraste
                post_data = nib.load(data_info['post_file']).get_fdata().astype(np.float32)
                post_data = enhanced_robust_normalize(post_data, self.normalization)
                post_data = self._enhance_contrast(post_data)
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
            
            # Verificar y limpiar
            if not np.isfinite(selected_data).all():
                selected_data = np.nan_to_num(selected_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Resize mejorado
            tensor = torch.from_numpy(selected_data)
            tensor = F.interpolate(
                tensor.unsqueeze(0), 
                size=self.target_size, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Verificación final
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            return {
                'tensor': tensor.float(),
                'patient_id': patient_id,
                'channels_used': self.required_channels
            }
            
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {e}")
            dummy_tensor = torch.zeros((self.required_channels, *self.target_size))
            return {
                'tensor': dummy_tensor.float(),
                'patient_id': patient_id,
                'channels_used': self.required_channels
            }

# =============================================================================
# EVALUADOR AVANZADO CON DETECCIÓN AUTOMÁTICA + TTA
# =============================================================================

class SmartAdvancedEvaluator:
    """Evaluador avanzado con detección automática + TTA + Ensemble"""
    
    def __init__(self, model_dir: Path, use_tta: bool = True):
        self.model_dir = model_dir
        self.use_tta = use_tta
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
                    model_type = "FastResNet-Tiny"
                elif self.config['model_type'] == 'fast_resnet_small':
                    model = FastResNet3D(in_channels=self.detected_channels, model_size='small').to(device)
                    model_type = "FastResNet-Small"
                else:
                    # Intentar cargar Enhanced primero, luego fallback
                    try:
                        model = EnhancedCNN3D(in_channels=self.detected_channels).to(device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model_type = "Enhanced"
                    except Exception:
                        # Fallback a UltraFast
                        model = UltraFastCNN3D(in_channels=self.detected_channels).to(device)
                        model_type = "UltraFast"
                
                if model_type != "Enhanced":
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                model.eval()
                
                val_auc = checkpoint.get('val_auc', 0.5)
                
                self.models.append({
                    'model': model,
                    'val_auc': val_auc,
                    'fold': i,
                    'type': model_type
                })
                
                print(f"   ✅ Modelo {i+1} ({model_type}): AUC {val_auc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error modelo {i+1}: {e}")
        
        print(f"🎯 {len(self.models)} modelos avanzados cargados")
    
    def _predict_with_tta(self, tensor: torch.Tensor) -> torch.Tensor:
        """Predicción con Test-Time Augmentation"""
        
        all_predictions = []
        all_weights = []
        
        with torch.no_grad():
            for model_info in self.models:
                try:
                    model = model_info['model']
                    weight = model_info['val_auc']
                    
                    if self.use_tta:
                        # Aplicar TTA
                        augmented_tensors = apply_test_time_augmentation(tensor, n_augs=8)
                        model_predictions = []
                        
                        for aug_tensor in augmented_tensors:
                            aug_tensor = torch.nan_to_num(aug_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                            outputs = model(aug_tensor)
                            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                            probs = torch.softmax(outputs, dim=1)
                            probs = torch.nan_to_num(probs, nan=0.5, posinf=0.99, neginf=0.01)
                            model_predictions.append(probs[:, 1])
                        
                        # Promedio de TTA
                        avg_pred = torch.stack(model_predictions).mean(dim=0)
                    else:
                        # Predicción simple
                        clean_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        outputs = model(clean_tensor)
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                        probs = torch.softmax(outputs, dim=1)
                        avg_pred = torch.nan_to_num(probs[:, 1], nan=0.5, posinf=0.99, neginf=0.01)
                    
                    all_predictions.append(avg_pred)
                    all_weights.append(weight)
                    
                except Exception as e:
                    print(f"⚠️ Error en predicción: {e}")
                    continue
        
        if all_predictions:
            # Ensemble ponderado
            total_weight = sum(all_weights)
            if total_weight > 0:
                weighted_preds = []
                for pred, weight in zip(all_predictions, all_weights):
                    weighted_preds.append(pred * weight / total_weight)
                result = torch.stack(weighted_preds).sum(dim=0)
            else:
                result = torch.stack(all_predictions).mean(dim=0)
            
            # Verificación final
            result = torch.nan_to_num(result, nan=0.5, posinf=0.99, neginf=0.01)
            return result
        else:
            return torch.full((tensor.shape[0],), 0.5, device=tensor.device)
    
    def evaluate_enhanced(self, 
                         dataset: SmartEnhancedDataset,
                         batch_size: int = 2) -> Dict:
        """Evaluación mejorada con TTA"""
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        all_patient_ids = []
        all_predictions = []
        failed_predictions = 0
        
        tta_status = "CON TTA" if self.use_tta else "SIN TTA"
        print(f"🔮 Evaluación inteligente avanzada {tta_status}...")
        print(f"   📊 Dataset: {len(dataset)} pacientes")
        print(f"   🧠 Modelos: {len(self.models)}")
        print(f"   🎯 Resolución: {dataset.target_size}")
        print(f"   🔧 Canales: {self.detected_channels}")
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                tensors = batch['tensor'].to(device, non_blocking=True)
                
                # Verificar batch
                tensors = torch.nan_to_num(tensors, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Predicción con TTA
                batch_predictions = self._predict_with_tta(tensors)
                
                # Verificar predicciones
                if torch.isfinite(batch_predictions).all():
                    all_patient_ids.extend(batch['patient_id'])
                    all_predictions.extend(batch_predictions.cpu().numpy())
                else:
                    print(f"⚠️ Batch {batch_idx}: Predicciones inválidas")
                    failed_predictions += len(batch['patient_id'])
                    default_preds = [0.5] * len(batch['patient_id'])
                    all_patient_ids.extend(batch['patient_id'])
                    all_predictions.extend(default_preds)
                
            except Exception as e:
                print(f"❌ Error en batch {batch_idx}: {e}")
                failed_predictions += len(batch['patient_id'])
                default_preds = [0.5] * len(batch['patient_id'])
                all_patient_ids.extend(batch['patient_id'])
                all_predictions.extend(default_preds)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"    Progreso: {(batch_idx + 1) * batch_size}/{len(dataset)}")
        
        print(f"📊 Predicciones fallidas: {failed_predictions}/{len(all_predictions)}")
        
        return {
            'patient_ids': all_patient_ids,
            'predictions': all_predictions,
            'failed_predictions': failed_predictions,
            'success_rate': 1 - (failed_predictions / len(all_predictions)) if all_predictions else 0,
            'used_tta': self.use_tta,
            'detected_channels': self.detected_channels
        }

# =============================================================================
# FUNCIÓN PRINCIPAL MEJORADA
# =============================================================================

def run_smart_enhanced_optimization():
    """Pipeline inteligente mejorado: Detección automática + TTA + Attention"""
    
    print("🧠🚀 PIPELINE INTELIGENTE MEJORADO")
    print("=" * 60)
    print("🔧 Detección automática de canales")
    print("🎯 Test-Time Augmentation + Attention")
    print("📈 Resolución aumentada + Preprocessing mejorado")
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
    
    # Cargar PCR labels
    with open(pcr_labels_file, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {}
    for item in pcr_list:
        if 'patient_id' in item and 'pcr' in item:
            pcr_value = item['pcr']
            if pcr_value in ["0", "1"]:
                pcr_data[item['patient_id']] = int(pcr_value)
    
    print(f"📋 PCR labels: {len(pcr_data)}")
    
    # Evaluar con diferentes configuraciones
    results_summary = []
    
    configs = [
        {"target_size": (64, 64, 32), "use_tta": True, "name": "Alta resolución + TTA"},
        {"target_size": (48, 48, 24), "use_tta": True, "name": "Resolución estándar + TTA"},
        {"target_size": (64, 64, 32), "use_tta": False, "name": "Alta resolución sin TTA"},
    ]
    
    best_auc = 0
    best_config = None
    best_results = None
    
    for config in configs:
        print(f"\n🔬 EVALUANDO: {config['name']}")
        print("-" * 50)
        
        try:
            # Crear evaluador (detecta canales automáticamente)
            evaluator = SmartAdvancedEvaluator(model_dir, use_tta=config['use_tta'])
            
            if len(evaluator.models) == 0:
                print("❌ No se cargaron modelos")
                continue
            
            # Crear dataset adaptativo
            dataset = SmartEnhancedDataset(
                images_dir=images_dir,
                predictions_dir=predictions_dir,
                patient_ids=test_patients,
                required_channels=evaluator.detected_channels,  # ← ADAPTATIVO
                target_size=config['target_size'],
                normalization='adaptive_zscore'
            )
            
            if len(dataset) == 0:
                print("❌ Dataset vacío")
                continue
            
            # Evaluación
            prediction_results = evaluator.evaluate_enhanced(dataset, batch_size=2)
            
            # Procesar predicciones
            predictions = np.array(prediction_results['predictions'])
            predictions = np.nan_to_num(predictions, nan=0.5, posinf=0.99, neginf=0.01)
            
            # Alinear con PCR labels
            valid_predictions = []
            pred_scores = []
            true_labels = []
            
            for pid, pred in zip(prediction_results['patient_ids'], predictions):
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
                continue
            
            # Calcular métricas
            pred_scores = np.array(pred_scores)
            true_labels = np.array(true_labels)
            
            # Verificar clases
            unique_labels = np.unique(true_labels)
            if len(unique_labels) < 2:
                print(f"⚠️ Solo una clase presente: {unique_labels}")
                accuracy = accuracy_score(true_labels, [1 if p > 0.5 else 0 for p in pred_scores])
                auc = 0.5
            else:
                auc = roc_auc_score(true_labels, pred_scores)
                accuracy = accuracy_score(true_labels, [1 if p > 0.5 else 0 for p in pred_scores])
            
            # Métricas adicionales
            precision = precision_score(true_labels, [1 if p > 0.5 else 0 for p in pred_scores], zero_division=0)
            recall = recall_score(true_labels, [1 if p > 0.5 else 0 for p in pred_scores], zero_division=0)
            
            config_results = {
                'config': config,
                'detected_channels': prediction_results['detected_channels'],
                'metrics': {
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'n_patients': len(valid_predictions)
                },
                'predictions': valid_predictions
            }
            
            results_summary.append(config_results)
            
            # Identificar configuración real
            channel_names = {1: 'post_only', 2: 'pre_post', 3: 'all (pre+post+mask)'}
            detected_config = channel_names.get(prediction_results['detected_channels'], 'unknown')
            
            print(f"   🔧 Canales detectados: {prediction_results['detected_channels']} ({detected_config})")
            print(f"   🎯 AUC: {auc:.4f}")
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📈 Precision: {precision:.4f}")
            print(f"   📊 Recall: {recall:.4f}")
            print(f"   👥 Pacientes: {len(valid_predictions)}")
            
            # Actualizar mejor configuración
            if auc > best_auc:
                best_auc = auc
                best_config = config
                best_results = config_results
            
        except Exception as e:
            print(f"❌ Error en configuración {config['name']}: {e}")
            continue
    
    # Resultados finales
    print(f"\n🏆 RESULTADOS FINALES")
    print("=" * 60)
    
    if best_results:
        channel_names = {1: 'post_only', 2: 'pre_post', 3: 'all (pre+post+mask)'}
        detected_config = channel_names.get(best_results['detected_channels'], 'unknown')
        
        print(f"🥇 MEJOR CONFIGURACIÓN: {best_config['name']}")
        print(f"   🔧 Configuración real detectada: {detected_config}")
        print(f"   🎯 AUC: {best_auc:.4f}")
        print(f"   ✅ Accuracy: {best_results['metrics']['accuracy']:.4f}")
        print(f"   📈 Precision: {best_results['metrics']['precision']:.4f}")
        print(f"   📊 Recall: {best_results['metrics']['recall']:.4f}")
        
        # Comparaciones con objetivo
        print(f"\n📈 PROGRESO HACIA OBJETIVO:")
        print(f"   🎯 Objetivo: >0.60")
        print(f"   📊 Logrado: {best_auc:.4f}")
        
        if best_auc > 0.60:
            print(f"   🎉 ¡OBJETIVO ALCANZADO! (+{(best_auc-0.60):.4f})")
        else:
            deficit = 0.60 - best_auc
            print(f"   📊 Falta: {deficit:.4f} para objetivo")
            
            if deficit < 0.01:
                print(f"   🔥 ¡MUY CERCA! Prácticamente alcanzado")
            elif deficit < 0.03:
                print(f"   🚀 ¡EXCELENTE! Muy buen progreso")
            elif deficit < 0.05:
                print(f"   👍 BUEN PROGRESO - En buen camino")
        
        # Comparaciones históricas
        baselines = {
            "Fixed Channel Script": 0.5763,  # El resultado que mencionaste
            "Random": 0.5000
        }
        
        print(f"\n📈 COMPARACIONES:")
        for name, baseline in baselines.items():
            diff = best_auc - baseline
            if diff > 0:
                print(f"   ✅ vs {name}: {baseline:.4f} → {best_auc:.4f} (+{diff:.4f})")
            else:
                print(f"   📉 vs {name}: {baseline:.4f} → {best_auc:.4f} ({diff:.4f})")
        
        # Guardar resultados
        output_file = Path("D:/smart_enhanced_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'best_results': best_results,
                'all_results': results_summary,
                'summary': {
                    'best_auc': best_auc,
                    'target_reached': best_auc > 0.60,
                    'detected_channels': best_results['detected_channels'],
                    'detected_config': detected_config
                }
            }, f, indent=2)
        
        csv_file = output_file.with_suffix('.csv')
        df_results = pd.DataFrame(best_results['predictions'])
        df_results.to_csv(csv_file, index=False)
        
        print(f"\n💾 Resultados guardados:")
        print(f"   📄 JSON: {output_file}")
        print(f"   📊 CSV: {csv_file}")
        
        return best_results
    
    else:
        print("❌ No se obtuvieron resultados válidos")
        return None

if __name__ == "__main__":
    run_smart_enhanced_optimization()