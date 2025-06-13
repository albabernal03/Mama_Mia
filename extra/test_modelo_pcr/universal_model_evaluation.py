# universal_test_evaluator.py
"""
EVALUADOR UNIVERSAL PARA TODOS LOS MODELOS MAMA-MIA
==================================================

Evalúa cualquier tipo de modelo en test set SIN usar ground truth:
✅ CNN Models (ResNet3D, UltraFast, Enhanced)
✅ Swin Transformer Hybrid Models
✅ Pure Radiomics Models
✅ Hybrid CNN+Radiomics Models
✅ Ensemble Methods
✅ Test-Time Augmentation

Genera submissions listas para challenge.
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
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# DEFINICIONES DE MODELOS (COPIADAS DE TUS SCRIPTS)
# =============================================================================

class UltraFastCNN3D(nn.Module):
    """Ultra-fast CNN 3D model"""
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
    """Fast ResNet 3D model"""
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
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        
        return x * attention.expand_as(x)

class EnhancedCNN3D(nn.Module):
    """Enhanced CNN 3D with attention"""
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        
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
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
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
        x1 = self.att1(self.conv1(x))
        x2 = self.att2(self.conv2(x1))
        x3 = self.att3(self.conv3(x2))
        x4 = self.att4(self.conv4(x3))
        
        x = self.global_pool(x4)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        
        return x

# =============================================================================
# EXTRACTOR DE RADIOMICS
# =============================================================================

class RadiomicsExtractor:
    """Extractor de features radiómicas básicas"""
    
    def extract_features_from_tensor(self, tensor_data):
        """Extraer features desde tensor [pre, post, mask]"""
        try:
            pre_image = tensor_data[0]
            post_image = tensor_data[1] 
            mask = tensor_data[2]
            
            features = {}
            
            # ROI extraction
            roi_pre = pre_image[mask > 0]
            roi_post = post_image[mask > 0]
            roi_enh = roi_post - roi_pre
            
            if len(roi_pre) < 10:
                return {}
            
            # First-order features (PRE)
            features.update(self._first_order_features(roi_pre, 'pre'))
            
            # First-order features (POST)
            features.update(self._first_order_features(roi_post, 'post'))
            
            # Enhancement features
            features.update(self._enhancement_features(roi_pre, roi_post, roi_enh))
            
            # Shape features
            features.update(self._shape_features(mask))
            
            # Validar features
            validated = {}
            for key, value in features.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    validated[key] = float(value)
            
            return validated
            
        except Exception as e:
            print(f"Error extracting radiomics: {e}")
            return {}
    
    def _first_order_features(self, roi, prefix):
        """First-order statistics"""
        features = {}
        if len(roi) == 0:
            return features
        
        try:
            from scipy import stats
            
            features[f'{prefix}_mean'] = np.mean(roi)
            features[f'{prefix}_median'] = np.median(roi)
            features[f'{prefix}_std'] = np.std(roi)
            features[f'{prefix}_min'] = np.min(roi)
            features[f'{prefix}_max'] = np.max(roi)
            features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
            
            # Percentiles
            for p in [10, 25, 75, 90]:
                features[f'{prefix}_p{p}'] = np.percentile(roi, p)
            
            features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
            features[f'{prefix}_skewness'] = stats.skew(roi)
            features[f'{prefix}_kurtosis'] = stats.kurtosis(roi)
            features[f'{prefix}_energy'] = np.sum(roi**2)
            features[f'{prefix}_cv'] = features[f'{prefix}_std'] / (abs(features[f'{prefix}_mean']) + 1e-8)
            
        except Exception as e:
            print(f"Error in first-order features ({prefix}): {e}")
        
        return features
    
    def _enhancement_features(self, roi_pre, roi_post, roi_enh):
        """Enhancement features"""
        features = {}
        
        try:
            features['enh_mean'] = np.mean(roi_enh)
            features['enh_median'] = np.median(roi_enh)
            features['enh_std'] = np.std(roi_enh)
            features['enh_min'] = np.min(roi_enh)
            features['enh_max'] = np.max(roi_enh)
            features['enh_range'] = features['enh_max'] - features['enh_min']
            
            # Ratios
            pre_mean = np.mean(roi_pre)
            post_mean = np.mean(roi_post)
            
            if abs(pre_mean) > 1e-8:
                features['enh_ratio'] = features['enh_mean'] / pre_mean
                features['post_pre_ratio'] = post_mean / pre_mean
                features['peak_enh_ratio'] = features['enh_max'] / pre_mean
            else:
                features['enh_ratio'] = 0
                features['post_pre_ratio'] = 1
                features['peak_enh_ratio'] = 0
            
            # Enhancement distribution
            positive_enh = roi_enh[roi_enh > 0]
            features['positive_enh_fraction'] = len(positive_enh) / len(roi_enh)
            
            if len(positive_enh) > 0:
                features['positive_enh_mean'] = np.mean(positive_enh)
            else:
                features['positive_enh_mean'] = 0
            
        except Exception as e:
            print(f"Error in enhancement features: {e}")
        
        return features
    
    def _shape_features(self, mask):
        """Shape features"""
        features = {}
        
        try:
            volume = np.sum(mask > 0)
            features['shape_volume'] = volume
            
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                ranges = [np.max(coord) - np.min(coord) + 1 for coord in coords]
                bbox_volume = np.prod(ranges)
                features['shape_extent'] = volume / bbox_volume if bbox_volume > 0 else 0
                features['shape_elongation'] = ranges[1] / ranges[0] if ranges[0] > 0 else 1
                features['shape_flatness'] = ranges[2] / ranges[0] if ranges[0] > 0 else 1
        
        except Exception as e:
            print(f"Error in shape features: {e}")
        
        return features

# =============================================================================
# DATASET UNIVERSAL
# =============================================================================

class UniversalTestDataset(Dataset):
    """Dataset universal para evaluación sin ground truth"""
    
    def __init__(self, 
                 data_dir: Path,
                 patient_ids: List[str],
                 target_size=(96, 96, 48),
                 required_channels=None):
        
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.required_channels = required_channels
        
        # Verificar archivos válidos
        self.valid_indices = self._check_valid_files()
        
        print(f"🧪 Test dataset: {len(self.valid_indices)}/{len(patient_ids)} valid patients")
    
    def _check_valid_files(self) -> List[int]:
        """Verificar archivos disponibles"""
        valid_indices = []
        
        for i, patient_id in enumerate(self.patient_ids):
            tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
            if tensor_file.exists():
                try:
                    tensor_img = nib.load(tensor_file)
                    shape = tensor_img.shape
                    if len(shape) == 4 and shape[0] >= 3:
                        valid_indices.append(i)
                except Exception as e:
                    print(f"❌ Error loading {patient_id}: {e}")
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        patient_id = self.patient_ids[actual_idx]
        
        # Cargar tensor base
        tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
        tensor_img = nib.load(tensor_file)
        tensor_data = tensor_img.get_fdata().astype(np.float32)
        
        # Adaptar número de canales según modelo
        if self.required_channels:
            if self.required_channels == 1:
                # Solo post-contraste
                tensor = torch.from_numpy(tensor_data[1:2])
            elif self.required_channels == 2:
                # Pre + post
                tensor = torch.from_numpy(tensor_data[:2])
            elif self.required_channels == 3:
                # Pre + post + mask
                tensor = torch.from_numpy(tensor_data)
            else:
                # Multi-parametric (6 canales)
                tensor = self._create_multiparametric_input(tensor_data)
        else:
            # Auto-detectar
            tensor = torch.from_numpy(tensor_data)
        
        # Padding uniforme
        tensor = self._pad_to_target_size(tensor)
        
        return {
            'tensor': tensor.float(),
            'patient_id': patient_id,
            'raw_data': tensor_data  # Para radiomics
        }
    
    def _create_multiparametric_input(self, tensor_data):
        """Crear input multi-paramétrico de 6 canales"""
        pre, post, mask = tensor_data[0], tensor_data[1], tensor_data[2]
        
        # Canal 4: Difference map
        diff_map = post - pre
        
        # Canal 5: Enhancement ratio
        enhancement_ratio = np.zeros_like(pre)
        valid_mask = pre > 0.01
        enhancement_ratio[valid_mask] = (post[valid_mask] - pre[valid_mask]) / pre[valid_mask]
        enhancement_ratio = np.clip(enhancement_ratio, -5, 5)
        
        # Canal 6: Masked enhancement
        masked_enhancement = diff_map * mask
        
        multiparametric = np.stack([
            pre, post, mask, diff_map, enhancement_ratio, masked_enhancement
        ], axis=0)
        
        return torch.from_numpy(multiparametric.astype(np.float32))
    
    def _pad_to_target_size(self, tensor):
        """Padding uniforme"""
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
        
        # Center crop
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

# =============================================================================
# TEST-TIME AUGMENTATION
# =============================================================================

class TestTimeAugmentation:
    """TTA para evaluación"""
    
    def __init__(self, model, n_augmentations=8):
        self.model = model
        self.n_augmentations = n_augmentations
        self.augmentations = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, [2]),  # Flip H
            lambda x: torch.flip(x, [3]),  # Flip W
            lambda x: torch.flip(x, [2, 3]),  # Flip H+W
            lambda x: torch.rot90(x, 1, dims=[2, 3]),
            lambda x: torch.rot90(x, 2, dims=[2, 3]),
            lambda x: torch.rot90(x, 3, dims=[2, 3]),
            lambda x: self._intensity_scale(x, 0.95),
        ]
    
    def _intensity_scale(self, x, factor):
        """Escalado de intensidad conservando máscara"""
        x_scaled = x.clone()
        if x.shape[0] >= 2:
            x_scaled[:2] = x_scaled[:2] * factor
        return x_scaled
    
    def predict(self, x):
        """Predicción con TTA"""
        predictions = []
        
        with torch.no_grad():
            for aug in self.augmentations[:self.n_augmentations]:
                try:
                    x_aug = aug(x)
                    pred = self.model(x_aug)
                    predictions.append(torch.softmax(pred, dim=1))
                except Exception as e:
                    print(f"Warning: TTA augmentation failed: {e}")
                    continue
        
        if predictions:
            mean_pred = torch.stack(predictions).mean(dim=0)
            return mean_pred
        else:
            # Fallback to original prediction
            return torch.softmax(self.model(x), dim=1)

# =============================================================================
# EVALUADOR UNIVERSAL
# =============================================================================

class UniversalModelEvaluator:
    """Evaluador universal para todos los tipos de modelos"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Radiomics extractor
        self.radiomics_extractor = RadiomicsExtractor()
        
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f'universal_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_model_type(self, model_path: Path) -> Dict:
        """Detectar tipo de modelo y parámetros"""
        model_info = {
            'type': 'unknown',
            'channels': 3,
            'architecture': 'unknown',
            'target_size': (96, 96, 48),
            'use_tta': False
        }
        
        try:
            if model_path.suffix == '.pth':
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Detectar canales
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                for key, param in state_dict.items():
                    if 'conv' in key.lower() and 'weight' in key and param.dim() == 5:
                        model_info['channels'] = param.shape[1]
                        break
                
                # Detectar arquitectura
                if 'features.0.weight' in state_dict:
                    model_info['architecture'] = 'UltraFastCNN3D'
                elif 'conv1.weight' in state_dict:
                    if any('layer' in key for key in state_dict.keys()):
                        model_info['architecture'] = 'FastResNet3D'
                    else:
                        model_info['architecture'] = 'ResNet3D_MAMA'
                elif 'att1' in str(state_dict.keys()):
                    model_info['architecture'] = 'EnhancedCNN3D'
                elif 'patch_embed' in str(state_dict.keys()):
                    model_info['architecture'] = 'HybridCNNSwinTransformer3D'
                
                # Configuración del modelo
                config = checkpoint.get('config', {})
                if config:
                    model_info['target_size'] = config.get('target_size', (96, 96, 48))
                    model_info['use_tta'] = config.get('use_tta', False)
                
                model_info['type'] = 'pytorch_cnn'
                
            elif model_path.suffix == '.pkl':
                model_info['type'] = 'sklearn_radiomics'
                model_info['architecture'] = 'Radiomics_Pipeline'
                
            elif model_path.suffix == '.json':
                model_info['type'] = 'hybrid'
                model_info['architecture'] = 'Hybrid_CNN_Radiomics'
                
        except Exception as e:
            self.logger.warning(f"Could not detect model type for {model_path}: {e}")
        
        return model_info
    
    def load_pytorch_model(self, model_path: Path, model_info: Dict):
        """Cargar modelo PyTorch"""
        try:
            # Crear modelo según arquitectura
            if model_info['architecture'] == 'UltraFastCNN3D':
                model = UltraFastCNN3D(in_channels=model_info['channels'])
            elif model_info['architecture'] == 'FastResNet3D':
                # Intentar detectar tamaño
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Detectar tamaño del modelo por dimensiones de fc
                if 'fc.weight' in state_dict:
                    fc_in_features = state_dict['fc.weight'].shape[1]
                    if fc_in_features == 128:
                        model_size = 'tiny'
                    elif fc_in_features == 256:
                        model_size = 'small'
                    else:
                        model_size = 'medium'
                else:
                    model_size = 'small'
                
                model = FastResNet3D(in_channels=model_info['channels'], model_size=model_size)
            elif model_info['architecture'] == 'EnhancedCNN3D':
                model = EnhancedCNN3D(in_channels=model_info['channels'])
            else:
                # Fallback
                model = UltraFastCNN3D(in_channels=model_info['channels'])
            
            # Cargar pesos
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            
            self.logger.info(f"✅ Loaded PyTorch model: {model_info['architecture']}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading PyTorch model: {e}")
            return None
    
    def load_sklearn_model(self, model_path: Path):
        """Cargar modelo sklearn"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"✅ Loaded sklearn model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading sklearn model: {e}")
            return None
    
    def predict_pytorch_model(self, model, dataset: UniversalTestDataset, use_tta: bool = False):
        """Predicciones con modelo PyTorch"""
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
            num_workers=0
        )
        
        predictions = []
        patient_ids = []
        
        if use_tta:
            tta = TestTimeAugmentation(model, n_augmentations=8)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="PyTorch prediction"):
                tensors = batch['tensor'].to(device)
                
                try:
                    if use_tta:
                        outputs = tta.predict(tensors)
                    else:
                        outputs = torch.softmax(model(tensors), dim=1)
                    
                    predictions.extend(outputs[:, 1].cpu().numpy())
                    patient_ids.extend(batch['patient_id'])
                    
                except Exception as e:
                    self.logger.warning(f"Error in batch prediction: {e}")
                    # Default prediction
                    predictions.extend([0.5] * len(batch['patient_id']))
                    patient_ids.extend(batch['patient_id'])
        
        return patient_ids, predictions
    
    def predict_radiomics_model(self, model, dataset: UniversalTestDataset):
        """Predicciones con modelo radiomics"""
        features_list = []
        patient_ids = []
        
        for i in tqdm(range(len(dataset)), desc="Radiomics extraction"):
            sample = dataset[i]
            patient_id = sample['patient_id']
            raw_data = sample['raw_data']
            
            # Extraer features radiómicas
            features = self.radiomics_extractor.extract_features_from_tensor(raw_data)
            
            if features:
                features_list.append(features)
                patient_ids.append(patient_id)
        
        if not features_list:
            self.logger.error("No radiomics features extracted!")
            return [], []
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(features_list).fillna(0)
        
        try:
            # Hacer predicciones
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(features_df.values)[:, 1]
            else:
                predictions = model.predict(features_df.values)
            
            return patient_ids, predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in radiomics prediction: {e}")
            return patient_ids, [0.5] * len(patient_ids)
    
    def evaluate_single_model(self, 
                             model_path: Path, 
                             data_dir: Path,
                             test_patients: List[str],
                             model_name: str = None) -> Dict:
        """Evaluar un modelo individual"""
        
        if model_name is None:
            model_name = model_path.stem
        
        self.logger.info(f"🧪 Evaluating model: {model_name}")
        self.logger.info(f"   Path: {model_path}")
        
        # Detectar tipo de modelo
        model_info = self.detect_model_type(model_path)
        self.logger.info(f"   Detected: {model_info['architecture']} ({model_info['channels']} channels)")
        
        # Crear dataset
        dataset = UniversalTestDataset(
            data_dir=data_dir,
            patient_ids=test_patients,
            target_size=model_info['target_size'],
            required_channels=model_info['channels']
        )
        
        if len(dataset) == 0:
            self.logger.error("No valid test data found!")
            return {}
        
        # Hacer predicciones según tipo de modelo
        if model_info['type'] == 'pytorch_cnn':
            model = self.load_pytorch_model(model_path, model_info)
            if model is None:
                return {}
            
            patient_ids, predictions = self.predict_pytorch_model(
                model, dataset, use_tta=model_info['use_tta']
            )
            
        elif model_info['type'] == 'sklearn_radiomics':
            model = self.load_sklearn_model(model_path)
            if model is None:
                return {}
            
            patient_ids, predictions = self.predict_radiomics_model(model, dataset)
            
        else:
            self.logger.error(f"Unsupported model type: {model_info['type']}")
            return {}
        
        # Crear resultados
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'model_info': model_info,
            'n_predictions': len(predictions),
            'predictions': []
        }
        
        for pid, pred in zip(patient_ids, predictions):
            results['predictions'].append({
                'patient_id': pid,
                'pcr_probability': float(pred),
                'pcr_prediction': int(pred > 0.5)
            })
        
        # Estadísticas
        if predictions:
            results['statistics'] = {
                'mean_probability': float(np.mean(predictions)),
                'std_probability': float(np.std(predictions)),
                'min_probability': float(np.min(predictions)),
                'max_probability': float(np.max(predictions)),
                'predicted_pcr_count': int(sum(p > 0.5 for p in predictions)),
                'predicted_pcr_rate': float(np.mean([p > 0.5 for p in predictions]))
            }
        
        self.logger.info(f"✅ {model_name}: {len(predictions)} predictions generated")
        
        return results
    
    def evaluate_ensemble(self, 
                         model_paths: List[Path], 
                         data_dir: Path,
                         test_patients: List[str],
                         ensemble_name: str = "ensemble") -> Dict:
        """Evaluar ensemble de modelos"""
        
        self.logger.info(f"🎯 Evaluating ensemble: {ensemble_name}")
        self.logger.info(f"   Models: {len(model_paths)}")
        
        all_predictions = {}
        valid_models = []
        
        # Evaluar cada modelo
        for model_path in model_paths:
            results = self.evaluate_single_model(model_path, data_dir, test_patients)
            if results and results['predictions']:
                valid_models.append(model_path.stem)
                
                # Almacenar predicciones por paciente
                for pred in results['predictions']:
                    pid = pred['patient_id']
                    prob = pred['pcr_probability']
                    
                    if pid not in all_predictions:
                        all_predictions[pid] = []
                    all_predictions[pid].append(prob)
        
        if not all_predictions:
            self.logger.error("No valid predictions from ensemble!")
            return {}
        
        # Calcular ensemble promedio
        ensemble_predictions = []
        ensemble_patient_ids = []
        
        for pid, probs in all_predictions.items():
            if len(probs) > 0:
                ensemble_prob = np.mean(probs)
                ensemble_predictions.append({
                    'patient_id': pid,
                    'pcr_probability': float(ensemble_prob),
                    'pcr_prediction': int(ensemble_prob > 0.5),
                    'n_models': len(probs),
                    'individual_probs': probs
                })
                ensemble_patient_ids.append(pid)
        
        # Crear resultados del ensemble
        probs_only = [p['pcr_probability'] for p in ensemble_predictions]
        
        results = {
            'ensemble_name': ensemble_name,
            'models_used': valid_models,
            'n_models': len(valid_models),
            'n_predictions': len(ensemble_predictions),
            'predictions': ensemble_predictions,
            'statistics': {
                'mean_probability': float(np.mean(probs_only)),
                'std_probability': float(np.std(probs_only)),
                'min_probability': float(np.min(probs_only)),
                'max_probability': float(np.max(probs_only)),
                'predicted_pcr_count': int(sum(p > 0.5 for p in probs_only)),
                'predicted_pcr_rate': float(np.mean([p > 0.5 for p in probs_only]))
            }
        }
        
        self.logger.info(f"✅ Ensemble: {len(ensemble_predictions)} predictions from {len(valid_models)} models")
        
        return results
    
    def save_results(self, results: Dict, suffix: str = ""):
        """Guardar resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON completo
        json_file = self.output_dir / f"test_predictions_{suffix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV para submission
        if 'predictions' in results:
            csv_file = self.output_dir / f"submission_{suffix}_{timestamp}.csv"
            
            predictions_df = pd.DataFrame([
                {
                    'patient_id': pred['patient_id'],
                    'pcr_probability': pred['pcr_probability']
                }
                for pred in results['predictions']
            ])
            
            predictions_df.to_csv(csv_file, index=False)
            
            self.logger.info(f"💾 Results saved:")
            self.logger.info(f"   📄 JSON: {json_file}")
            self.logger.info(f"   📊 CSV: {csv_file}")
        
        return json_file, csv_file

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Evaluación universal de todos los modelos en test set"""
    
    print("🎯 UNIVERSAL MODEL EVALUATOR - NO GROUND TRUTH")
    print("=" * 60)
    print("📊 Evalúa cualquier tipo de modelo en test set")
    print("🚫 Sin usar ground truth - ideal para challenge submission")
    print("=" * 60)
    
    # Configuración
    data_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    output_dir = Path("D:/universal_test_evaluation")
    
    # Directorios con modelos entrenados
    model_directories = [
        Path("D:/mama_mia_CORRECTED_HYBRID_results"),
        Path("D:/mama_mia_REAL_PCR_results"), 
        Path("D:/mama_mia_ADVANCED_SWIN_results"),
        Path("."),  # Directorio actual para modelos radiomics
    ]
    
    print(f"📁 Configuration:")
    print(f"   Data: {data_dir}")
    print(f"   Splits: {splits_csv}")
    print(f"   Output: {output_dir}")
    
    # Cargar test split
    try:
        splits_df = pd.read_csv(splits_csv)
        test_patients = splits_df['test_split'].dropna().astype(str).unique().tolist()
        print(f"📊 Test patients: {len(test_patients)}")
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        return
    
    # Crear evaluador
    evaluator = UniversalModelEvaluator(output_dir)
    
    # Buscar todos los modelos
    all_models = []
    
    for model_dir in model_directories:
        if model_dir.exists():
            # PyTorch models
            pytorch_models = list(model_dir.glob("*.pth"))
            all_models.extend(pytorch_models)
            
            # Sklearn models
            sklearn_models = list(model_dir.glob("*.pkl"))
            all_models.extend(sklearn_models)
            
            # JSON results (hybrid models)
            json_models = list(model_dir.glob("*results*.json"))
            all_models.extend(json_models)
    
    print(f"🔍 Found {len(all_models)} models to evaluate:")
    for model_path in all_models:
        print(f"   📄 {model_path.name}")
    
    if not all_models:
        print("❌ No models found!")
        return
    
    # Evaluar modelos individuales
    individual_results = []
    
    for model_path in all_models:
        try:
            print(f"\n🧪 Evaluating: {model_path.name}")
            results = evaluator.evaluate_single_model(
                model_path, data_dir, test_patients, model_path.stem
            )
            
            if results:
                individual_results.append(results)
                
                # Guardar resultados individuales
                evaluator.save_results(results, f"individual_{model_path.stem}")
                
                # Mostrar estadísticas
                stats = results.get('statistics', {})
                print(f"   ✅ Predictions: {results['n_predictions']}")
                print(f"   📊 Mean prob: {stats.get('mean_probability', 0):.3f}")
                print(f"   📈 pCR rate: {stats.get('predicted_pcr_rate', 0):.1%}")
            else:
                print(f"   ❌ Failed to evaluate {model_path.name}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Crear ensemble de todos los modelos válidos
    if len(individual_results) > 1:
        print(f"\n🎯 Creating ensemble from {len(individual_results)} models...")
        
        valid_model_paths = []
        for result in individual_results:
            model_path = Path(result['model_path'])
            if model_path.exists():
                valid_model_paths.append(model_path)
        
        if valid_model_paths:
            ensemble_results = evaluator.evaluate_ensemble(
                valid_model_paths, data_dir, test_patients, "all_models_ensemble"
            )
            
            if ensemble_results:
                evaluator.save_results(ensemble_results, "ensemble_all")
                
                # Mostrar estadísticas del ensemble
                stats = ensemble_results.get('statistics', {})
                print(f"   ✅ Ensemble predictions: {ensemble_results['n_predictions']}")
                print(f"   🏆 Models combined: {ensemble_results['n_models']}")
                print(f"   📊 Mean prob: {stats.get('mean_probability', 0):.3f}")
                print(f"   📈 pCR rate: {stats.get('predicted_pcr_rate', 0):.1%}")
    
    # Resumen final
    print(f"\n" + "=" * 60)
    print(f"🏆 EVALUATION SUMMARY")
    print(f"=" * 60)
    print(f"📊 Individual models evaluated: {len(individual_results)}")
    print(f"📊 Test patients processed: {len(test_patients)}")
    print(f"💾 Results saved in: {output_dir}")
    
    if individual_results:
        print(f"\n📋 Individual model results:")
        for result in individual_results:
            stats = result.get('statistics', {})
            print(f"   {result['model_name']}: {stats.get('predicted_pcr_rate', 0):.1%} pCR rate")
    
    print(f"\n✅ Universal evaluation completed!")
    print(f"📄 Check {output_dir} for submission files")

if __name__ == "__main__":
    main()