r"""
MAMA-MIA TEST INFERENCE - Validación en conjunto de test
🎯 Evaluar el modelo optimizado en datos no vistos
📊 Validación real del AUC 0.8592 obtenido en train
🏆 Preparación para submission al challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import nibabel as nib
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

def set_random_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Copiar las clases del modelo optimizado
class OptimizedTemporalDataset(Dataset):
    """Dataset para inferencia en test"""
    
    def __init__(self, patient_ids, data_dir, target_size=(72, 72, 36)):
        self.patient_ids = patient_ids
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        try:
            tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
            tensor_img = nib.load(tensor_file)
            tensor_data = tensor_img.get_fdata().astype(np.float32)
            
            pre_image = tensor_data[0]
            post_image = tensor_data[1]
            mask = tensor_data[2]
            
            # Resize
            pre_resized = self._resize_to_target(pre_image)
            post_resized = self._resize_to_target(post_image)
            mask_resized = self._resize_to_target(mask)
            
            # Simulación temporal (igual que en entrenamiento)
            timepoints = self._enhanced_temporal_sequence(pre_resized, post_resized, mask_resized)
            
            temporal_stack = np.stack([
                timepoints['baseline'],
                timepoints['early'], 
                timepoints['late']
            ], axis=0)
            
            return {
                'temporal_data': torch.FloatTensor(temporal_stack),
                'patient_id': patient_id
            }
            
        except Exception as e:
            print(f"Error loading {patient_id}: {e}")
            return {
                'temporal_data': torch.zeros((3, *self.target_size)),
                'patient_id': patient_id
            }
    
    def _resize_to_target(self, image):
        current_shape = image.shape
        zoom_factors = [self.target_size[i] / current_shape[i] for i in range(3)]
        resized = zoom(image, zoom_factors, order=1)
        return resized
    
    def _enhanced_temporal_sequence(self, pre_image, post_image, mask):
        """Misma simulación temporal que en entrenamiento"""
        enhancement = post_image - pre_image
        
        if np.sum(mask) > 0:
            enh_roi = enhancement[mask > 0]
            enh_mean = np.mean(enh_roi)
            enh_std = np.std(enh_roi)
            enh_ratio = enh_std / (abs(enh_mean) + 1e-6)
        else:
            enh_mean = np.mean(enhancement)
            enh_ratio = 1.0
        
        enhancement_threshold = np.percentile(enhancement.flatten(), 70)
        
        if enh_mean > enhancement_threshold and enh_ratio < 1.8:
            early_reduction = np.random.uniform(0.70, 0.85)
            late_reduction = np.random.uniform(0.45, 0.65)
        elif enh_mean > np.percentile(enhancement.flatten(), 50):
            early_reduction = np.random.uniform(0.40, 0.60)
            late_reduction = np.random.uniform(0.25, 0.45)
        else:
            early_reduction = np.random.uniform(0.10, 0.30)
            late_reduction = np.random.uniform(0.05, 0.20)
        
        base_snr = np.random.normal(1.0, 0.025)
        early_snr = np.random.normal(1.0, 0.030)
        late_snr = np.random.normal(1.0, 0.035)
        
        # Spatial heterogeneity
        if np.sum(mask) > 0:
            coords = np.where(mask > 0)
            if len(coords[0]) > 10:
                center_z = np.mean(coords[0])
                center_y = np.mean(coords[1]) 
                center_x = np.mean(coords[2])
                
                z_grid, y_grid, x_grid = np.meshgrid(
                    np.arange(mask.shape[0]), 
                    np.arange(mask.shape[1]), 
                    np.arange(mask.shape[2]), 
                    indexing='ij'
                )
                
                dist_from_center = np.sqrt(
                    ((z_grid - center_z) / mask.shape[0])**2 + 
                    ((y_grid - center_y) / mask.shape[1])**2 + 
                    ((x_grid - center_x) / mask.shape[2])**2
                )
                
                spatial_factor = 1.0 - 0.3 * dist_from_center
                spatial_factor = np.clip(spatial_factor, 0.3, 1.0)
            else:
                spatial_factor = 1.0
        else:
            spatial_factor = 1.0
        
        early_reduction_map = early_reduction * spatial_factor
        late_reduction_map = late_reduction * spatial_factor
        
        timepoints = {
            'baseline': post_image * base_snr,
            'early': post_image * (1 - early_reduction_map * early_snr),
            'late': post_image * (1 - late_reduction_map * late_snr)
        }
        
        return timepoints

class EnhancedTemporal3DCNN(nn.Module):
    """Misma arquitectura que en entrenamiento"""
    
    def __init__(self, input_channels=3, num_classes=2, dropout=0.3):
        super(EnhancedTemporal3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)
        
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        
        self.attention = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout/2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        attention = self.attention(x)
        x = x * attention
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class OptimizedRadiomicsExtractor:
    """Extractor de radiomics optimizado"""
    
    def __init__(self):
        pass
    
    def extract_delta_radiomics_features(self, pre_image, post_image, mask):
        """Delta radiomics features"""
        features = {}
        
        try:
            if mask is None or np.sum(mask) == 0:
                return self._get_default_delta_features()
            
            delta_image = post_image - pre_image
            roi_pre = pre_image[mask > 0]
            roi_post = post_image[mask > 0]
            roi_delta = delta_image[mask > 0]
            
            if len(roi_delta) == 0:
                return self._get_default_delta_features()
            
            # Intensity changes
            features['delta_mean_intensity'] = np.mean(roi_delta)
            features['delta_std_intensity'] = np.std(roi_delta)
            features['delta_median_intensity'] = np.median(roi_delta)
            features['delta_p90_intensity'] = np.percentile(roi_delta, 90)
            features['delta_p10_intensity'] = np.percentile(roi_delta, 10)
            features['delta_iqr_intensity'] = np.percentile(roi_delta, 75) - np.percentile(roi_delta, 25)
            
            # Enhancement ratios
            pre_mean = np.mean(roi_pre) + 1e-6
            features['enhancement_ratio_mean'] = np.mean(roi_delta) / pre_mean
            features['enhancement_ratio_max'] = np.max(roi_delta) / pre_mean
            features['enhancement_ratio_median'] = np.median(roi_delta) / pre_mean
            features['enhancement_ratio_std'] = np.std(roi_delta) / pre_mean
            
            # Heterogeneity changes
            features['delta_heterogeneity'] = np.std(roi_delta) / (abs(np.mean(roi_delta)) + 1e-6)
            features['pre_heterogeneity'] = np.std(roi_pre) / (np.mean(roi_pre) + 1e-6)
            features['post_heterogeneity'] = np.std(roi_post) / (np.mean(roi_post) + 1e-6)
            features['heterogeneity_change'] = features['post_heterogeneity'] - features['pre_heterogeneity']
            
            # Volume changes
            strong_enhancement_threshold = np.percentile(roi_delta, 75)
            weak_enhancement_threshold = np.percentile(roi_delta, 25)
            
            features['responding_volume_fraction'] = np.sum(roi_delta > strong_enhancement_threshold) / len(roi_delta)
            features['non_responding_volume_fraction'] = np.sum(roi_delta < weak_enhancement_threshold) / len(roi_delta)
            features['stable_volume_fraction'] = 1 - features['responding_volume_fraction'] - features['non_responding_volume_fraction']
            
            # Kinetic patterns
            enhancement_velocity = roi_delta / (roi_pre + 1e-6)
            features['kinetic_velocity_mean'] = np.mean(enhancement_velocity)
            features['kinetic_velocity_std'] = np.std(enhancement_velocity)
            features['kinetic_velocity_max'] = np.max(enhancement_velocity)
            features['kinetic_velocity_p90'] = np.percentile(enhancement_velocity, 90)
            
            # Spatial patterns
            try:
                coords = np.where(mask > 0)
                if len(coords[0]) > 20:
                    center_z = np.mean(coords[0])
                    center_y = np.mean(coords[1])
                    center_x = np.mean(coords[2])
                    
                    z_coords, y_coords, x_coords = coords
                    distances = np.sqrt(
                        (z_coords - center_z)**2 + 
                        (y_coords - center_y)**2 + 
                        (x_coords - center_x)**2
                    )
                    
                    distance_threshold = np.percentile(distances, 60)
                    central_mask = distances <= distance_threshold
                    peripheral_mask = distances > distance_threshold
                    
                    if np.sum(central_mask) > 0 and np.sum(peripheral_mask) > 0:
                        central_enhancement = roi_delta[central_mask]
                        peripheral_enhancement = roi_delta[peripheral_mask]
                        
                        features['central_enhancement_mean'] = np.mean(central_enhancement)
                        features['peripheral_enhancement_mean'] = np.mean(peripheral_enhancement)
                        features['central_peripheral_ratio'] = features['central_enhancement_mean'] / (features['peripheral_enhancement_mean'] + 1e-6)
                        features['enhancement_rim_pattern'] = features['peripheral_enhancement_mean'] - features['central_enhancement_mean']
                    else:
                        features.update(self._get_default_spatial_features())
                else:
                    features.update(self._get_default_spatial_features())
            except Exception:
                features.update(self._get_default_spatial_features())
            
            return features
            
        except Exception as e:
            print(f"Error in delta features: {e}")
            return self._get_default_delta_features()
    
    def _get_default_spatial_features(self):
        return {
            'central_enhancement_mean': 0.0, 'peripheral_enhancement_mean': 0.0,
            'central_peripheral_ratio': 1.0, 'enhancement_rim_pattern': 0.0
        }
    
    def _get_default_delta_features(self):
        features = {
            'delta_mean_intensity': 0.0, 'delta_std_intensity': 0.0, 'delta_median_intensity': 0.0,
            'delta_p90_intensity': 0.0, 'delta_p10_intensity': 0.0, 'delta_iqr_intensity': 0.0,
            'enhancement_ratio_mean': 0.0, 'enhancement_ratio_max': 0.0, 'enhancement_ratio_median': 0.0,
            'enhancement_ratio_std': 0.0, 'delta_heterogeneity': 0.0, 'pre_heterogeneity': 0.0,
            'post_heterogeneity': 0.0, 'heterogeneity_change': 0.0, 'responding_volume_fraction': 0.0,
            'non_responding_volume_fraction': 0.0, 'stable_volume_fraction': 1.0, 'kinetic_velocity_mean': 0.0,
            'kinetic_velocity_std': 0.0, 'kinetic_velocity_max': 0.0, 'kinetic_velocity_p90': 0.0
        }
        features.update(self._get_default_spatial_features())
        return features
    
    def extract_comprehensive_features(self, patient_id, data_dir):
        """Extract features para test"""
        patient_dir = Path(data_dir) / patient_id
        tensor_file = patient_dir / f"{patient_id}_tensor_3ch.nii.gz"
        
        if not tensor_file.exists():
            return self._get_all_default_features()
        
        try:
            tensor_img = nib.load(tensor_file)
            tensor_data = tensor_img.get_fdata().astype(np.float32)
            
            if tensor_data.ndim != 4 or tensor_data.shape[0] < 3:
                return self._get_all_default_features()
            
            pre_image = tensor_data[0]
            post_image = tensor_data[1] 
            mask = tensor_data[2]
            
            all_features = {}
            
            # Delta features
            delta_features = self.extract_delta_radiomics_features(pre_image, post_image, mask)
            all_features.update(delta_features)
            
            # Basic features
            if mask is not None and np.sum(mask) > 0:
                roi_post = post_image[mask > 0]
                roi_pre = pre_image[mask > 0]
                
                if len(roi_post) > 0:
                    all_features.update({
                        'post_mean': np.mean(roi_post),
                        'post_std': np.std(roi_post),
                        'post_median': np.median(roi_post),
                        'post_p90': np.percentile(roi_post, 90),
                        'post_p10': np.percentile(roi_post, 10),
                        'pre_mean': np.mean(roi_pre),
                        'pre_std': np.std(roi_pre),
                        'pre_median': np.median(roi_pre),
                        'volume': np.sum(mask),
                        'tumor_intensity_ratio': np.mean(roi_post) / (np.mean(roi_pre) + 1e-6)
                    })
                else:
                    all_features.update(self._get_basic_default_features())
            else:
                all_features.update(self._get_basic_default_features())
            
            # Validate features
            for key, value in all_features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    all_features[key] = 0.0
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting features for {patient_id}: {e}")
            return self._get_all_default_features()
    
    def _get_basic_default_features(self):
        return {
            'post_mean': 0.0, 'post_std': 0.0, 'post_median': 0.0,
            'post_p90': 0.0, 'post_p10': 0.0, 'pre_mean': 0.0,
            'pre_std': 0.0, 'pre_median': 0.0, 'volume': 0.0,
            'tumor_intensity_ratio': 1.0
        }
    
    def _get_all_default_features(self):
        all_features = {}
        all_features.update(self._get_default_delta_features())
        all_features.update(self._get_basic_default_features())
        return all_features

class MAMAMIATestInference:
    """
    🎯 Inferencia en conjunto de test
    """
    
    def __init__(self, data_dir, splits_csv, pcr_labels_file, device='cuda'):
        self.data_dir = Path(data_dir)
        self.splits_csv = splits_csv
        self.pcr_labels_file = pcr_labels_file
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load data
        self.splits_df = pd.read_csv(splits_csv)
        with open(pcr_labels_file, 'r') as f:
            pcr_list = json.load(f)
        
        self.pcr_data = {}
        for item in pcr_list:
            patient_id = item['patient_id']
            self.pcr_data[patient_id] = item
        
        self.radiomics_extractor = OptimizedRadiomicsExtractor()
        self.test_patients, self.test_labels = self._prepare_test_data()
        
        print(f"🎯 MAMA-MIA TEST INFERENCE")
        print(f"   Device: {self.device}")
        print(f"   Test patients: {len(self.test_patients)}")
        print(f"   Test pCR rate: {sum(self.test_labels)/len(self.test_labels):.1%}")
    
    def _prepare_test_data(self):
        test_patients = self.splits_df['test_split'].dropna().tolist()
        test_labels = []
        valid_patients = []
        
        for patient_id in test_patients:
            if patient_id in self.pcr_data:
                try:
                    pcr_status = self.pcr_data[patient_id]['pcr']
                    if pcr_status != 'unknown':
                        test_labels.append(int(pcr_status))
                        valid_patients.append(patient_id)
                except KeyError:
                    continue
        
        return valid_patients, test_labels
    
    def load_trained_models(self):
        """Cargar modelos entrenados"""
        print("📂 Cargando modelos entrenados...")
        
        # Cargar Temporal CNN
        temporal_model = None
        if Path('best_optimized_temporal_cnn.pth').exists():
            try:
                temporal_model = EnhancedTemporal3DCNN(input_channels=3, num_classes=2, dropout=0.3).to(self.device)
                temporal_model.load_state_dict(torch.load('best_optimized_temporal_cnn.pth', map_location=self.device))
                temporal_model.eval()
                print("✅ Temporal CNN cargado exitosamente")
            except Exception as e:
                print(f"❌ Error cargando Temporal CNN: {e}")
                temporal_model = None
        else:
            print("❌ No encontrado: best_optimized_temporal_cnn.pth")
        
        # Cargar ensemble (si existe)
        ensemble_model = None
        if Path('best_ensemble_model.pkl').exists():
            try:
                with open('best_ensemble_model.pkl', 'rb') as f:
                    ensemble_model = pickle.load(f)
                print("✅ Ensemble model cargado exitosamente")
            except Exception as e:
                print(f"❌ Error cargando ensemble: {e}")
                ensemble_model = None
        else:
            print("⚠️ No encontrado: best_ensemble_model.pkl")
        
        return temporal_model, ensemble_model
    
    def extract_test_features(self):
        """Extraer features del conjunto de test"""
        print("🔄 Extrayendo features del conjunto de test...")
        all_features = []
        
        for patient_id in tqdm(self.test_patients, desc="Extracting test features"):
            features = self.radiomics_extractor.extract_comprehensive_features(patient_id, self.data_dir)
            if features is not None and len(features) > 0:
                all_features.append(features)
            else:
                default_features = self.radiomics_extractor._get_all_default_features()
                all_features.append(default_features)
        
        features_df = pd.DataFrame(all_features).fillna(0)
        print(f"✅ Extraídas {len(features_df.columns)} features para test")
        
        return features_df
    
    def predict_temporal_cnn(self, temporal_model):
        """Predicciones con Temporal CNN"""
        if temporal_model is None:
            return None
        
        print("🚀 Generando predicciones con Temporal CNN...")
        
        # Create test dataset
        test_dataset = OptimizedTemporalDataset(
            self.test_patients,
            self.data_dir,
            target_size=(72, 72, 36)
        )
        
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        temporal_model.eval()
        predictions = []
        patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="CNN Inference"):
                try:
                    temporal_data = batch['temporal_data'].to(self.device)
                    batch_patient_ids = batch['patient_id']
                    
                    outputs = temporal_model(temporal_data)
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    
                    predictions.extend(probs.cpu().numpy())
                    patient_ids.extend(batch_patient_ids)
                    
                except Exception as e:
                    print(f"Error en batch: {e}")
                    # Add default predictions for failed batches
                    batch_size = len(batch['patient_id'])
                    predictions.extend([0.5] * batch_size)
                    patient_ids.extend(batch['patient_id'])
        
        return np.array(predictions), patient_ids
    
    def predict_ensemble(self, ensemble_model, features_df):
        """Predicciones con ensemble"""
        if ensemble_model is None:
            return None
        
        print("🔥 Generando predicciones con Ensemble...")
        
        try:
            # Cargar scaler si existe
            scaler = None
            if Path('feature_scaler.pkl').exists():
                with open('feature_scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(features_df.values)
            else:
                print("⚠️ No se encontró scaler, usando features sin escalar")
                X_scaled = features_df.values
            
            # Hacer predicciones
            if hasattr(ensemble_model, 'predict_proba'):
                predictions = ensemble_model.predict_proba(X_scaled)[:, 1]
            else:
                predictions = ensemble_model.decision_function(X_scaled)
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            return predictions
            
        except Exception as e:
            print(f"Error en ensemble prediction: {e}")
            return None
    
    def run_test_inference(self):
        """
        🎯 Ejecutar inferencia completa en test
        """
        print("🎯 MAMA-MIA TEST INFERENCE - VALIDACIÓN REAL")
        print("="*70)
        print("📊 Evaluando modelo optimizado en conjunto de test")
        print("🏆 Validación del AUC 0.8592 obtenido en train")
        print("="*70)
        
        # 1. Cargar modelos entrenados
        temporal_model, ensemble_model = self.load_trained_models()
        
        if temporal_model is None and ensemble_model is None:
            print("❌ No se encontraron modelos entrenados")
            return None
        
        # 2. Extraer features para test
        features_df = self.extract_test_features()
        
        # 3. Generar predicciones
        temporal_predictions = None
        ensemble_predictions = None
        
        # Temporal CNN predictions
        if temporal_model is not None:
            temporal_preds, patient_ids = self.predict_temporal_cnn(temporal_model)
            temporal_predictions = temporal_preds
        
        # Ensemble predictions
        if ensemble_model is not None:
            ensemble_predictions = self.predict_ensemble(ensemble_model, features_df)
        
        # 4. Combinar predicciones (igual que en entrenamiento)
        if temporal_predictions is not None and ensemble_predictions is not None:
            # Combinación inteligente
            temporal_auc_train = 0.8105  # Del entrenamiento
            ensemble_auc_train = 0.5715  # Del entrenamiento
            
            if temporal_auc_train > ensemble_auc_train:
                temporal_weight = 0.75
                ensemble_weight = 0.25
            else:
                temporal_weight = 0.6
                ensemble_weight = 0.4
            
            final_predictions = temporal_predictions * temporal_weight + ensemble_predictions * ensemble_weight
            print(f"🔄 Combinando predicciones: Temporal ({temporal_weight:.2f}) + Ensemble ({ensemble_weight:.2f})")
            
        elif temporal_predictions is not None:
            final_predictions = temporal_predictions
            print("🚀 Usando solo Temporal CNN")
            
        elif ensemble_predictions is not None:
            final_predictions = ensemble_predictions
            print("🔥 Usando solo Ensemble")
            
        else:
            print("❌ No se pudieron generar predicciones")
            return None
        
        # 5. Evaluar rendimiento en test
        test_auc = roc_auc_score(self.test_labels, final_predictions)
        test_balanced_acc = balanced_accuracy_score(self.test_labels, (final_predictions > 0.5).astype(int))
        
        # Métricas adicionales
        y_pred_binary = (final_predictions > 0.5).astype(int)
        cm = confusion_matrix(self.test_labels, y_pred_binary)
        
        # Calcular especificidad y sensibilidad
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 6. Resultados finales
        print("\n" + "="*70)
        print("🎯 RESULTADOS TEST - VALIDACIÓN REAL")
        print("="*70)
        print(f"🏆 AUC en TEST: {test_auc:.4f}")
        print(f"📊 Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"🎯 Sensibilidad (Recall): {sensitivity:.4f}")
        print(f"🎯 Especificidad: {specificity:.4f}")
        
        # Comparación con entrenamiento
        train_auc = 0.8592  # Tu resultado en train
        generalization_gap = train_auc - test_auc
        
        print(f"\n📈 ANÁLISIS DE GENERALIZACIÓN:")
        print(f"   AUC Training: {train_auc:.4f}")
        print(f"   AUC Test: {test_auc:.4f}")
        print(f"   Generalization Gap: {generalization_gap:.4f}")
        
        if generalization_gap < 0.05:
            print("✅ EXCELENTE generalización (gap < 0.05)")
        elif generalization_gap < 0.10:
            print("✅ BUENA generalización (gap < 0.10)")
        elif generalization_gap < 0.15:
            print("⚠️ Generalización moderada (gap < 0.15)")
        else:
            print("❌ Posible overfitting (gap > 0.15)")
        
        # Matriz de confusión
        print(f"\n📊 MATRIZ DE CONFUSIÓN:")
        print(f"   TN: {tn}, FP: {fp}")
        print(f"   FN: {fn}, TP: {tp}")
        
        # Análisis por percentiles de confianza
        print(f"\n🔍 ANÁLISIS DE CONFIANZA:")
        p25, p50, p75 = np.percentile(final_predictions, [25, 50, 75])
        print(f"   P25: {p25:.3f}, P50: {p50:.3f}, P75: {p75:.3f}")
        
        # Pacientes con alta confianza
        high_conf_positive = np.sum(final_predictions > 0.8)
        high_conf_negative = np.sum(final_predictions < 0.2)
        uncertain = np.sum((final_predictions >= 0.4) & (final_predictions <= 0.6))
        
        print(f"   Alta confianza pCR (>0.8): {high_conf_positive}")
        print(f"   Alta confianza no-pCR (<0.2): {high_conf_negative}")
        print(f"   Inciertos (0.4-0.6): {uncertain}")
        
        # Assessment final
        print(f"\n🎯 ASSESSMENT FINAL:")
        if test_auc >= 0.85:
            print("🏆 EXCELENTE rendimiento en test!")
            print("✅ Modelo listo para submission")
        elif test_auc >= 0.80:
            print("🎊 MUY BUEN rendimiento en test!")
            print("✅ Modelo competitivo para challenge")
        elif test_auc >= 0.75:
            print("✅ BUEN rendimiento en test")
            print("💪 Modelo sólido con margen de mejora")
        elif test_auc >= 0.70:
            print("📈 Rendimiento moderado en test")
            print("🔧 Necesita optimización adicional")
        else:
            print("⚠️ Rendimiento bajo en test")
            print("🔨 Requiere revisión del modelo")
        
        # Preparar datos para submission
        results = {
            'test_auc': float(test_auc),
            'test_balanced_accuracy': float(test_balanced_acc),
            'test_sensitivity': float(sensitivity),
            'test_specificity': float(specificity),
            'train_auc': float(train_auc),
            'generalization_gap': float(generalization_gap),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 
                'fn': int(fn), 'tp': int(tp)
            },
            'confidence_stats': {
                'p25': float(p25), 'p50': float(p50), 'p75': float(p75),
                'high_conf_positive': int(high_conf_positive),
                'high_conf_negative': int(high_conf_negative),
                'uncertain': int(uncertain)
            },
            'test_patients_count': len(self.test_patients),
            'test_pcr_rate': float(sum(self.test_labels)/len(self.test_labels)),
            'model_components_used': {
                'temporal_cnn': temporal_model is not None,
                'ensemble': ensemble_model is not None,
                'combination_used': temporal_predictions is not None and ensemble_predictions is not None
            }
        }
        
        # Crear submission file
        submission_df = pd.DataFrame({
            'patient_id': self.test_patients,
            'pcr_probability': final_predictions,
            'pcr_prediction': (final_predictions > 0.5).astype(int),
            'true_label': self.test_labels
        })
        
        # Guardar resultados
        submission_df.to_csv('mama_mia_test_results.csv', index=False)
        
        # Crear archivo para submission al challenge (sin true labels)
        challenge_submission = submission_df[['patient_id', 'pcr_probability']].copy()
        challenge_submission.to_csv('mama_mia_challenge_submission.csv', index=False)
        
        with open('mama_mia_test_inference_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 ARCHIVOS GENERADOS:")
        print(f"   📊 mama_mia_test_results.csv (resultados completos)")
        print(f"   🏆 mama_mia_challenge_submission.csv (para challenge)")
        print(f"   📈 mama_mia_test_inference_results.json (métricas)")
        
        # Recomendaciones finales
        print(f"\n🚀 RECOMENDACIONES:")
        if test_auc >= 0.85:
            print("   1. ✅ SUBMIT al challenge inmediatamente")
            print("   2. 🏆 Excelente rendimiento confirmado")
            print("   3. 💰 Competitivo para premio")
        elif test_auc >= 0.80:
            print("   1. ✅ Submit al challenge")
            print("   2. 💪 Muy buen modelo validado")
            print("   3. 🎯 Posibilidad real de premio")
        elif test_auc >= 0.75:
            print("   1. 🎯 Considerar submission")
            print("   2. 🔧 Optimizar si hay tiempo")
            print("   3. 📊 Analizar errores para mejorar")
        else:
            print("   1. 🔧 Optimizar antes de submission")
            print("   2. 🔍 Revisar overfitting")
            print("   3. 📈 Mejorar generalización")
        
        return results

def main():
    """
    🎯 Ejecutar inferencia en test
    """
    
    set_random_seeds(42)
    
    # Paths (ajusta a tu configuración)
    data_dir = r"D:\mama_mia_final_corrected"
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    pcr_labels_file = r"D:\clinical_data_complete.json"
    
    print("🎯 MAMA-MIA TEST INFERENCE")
    print("="*70)
    print("📊 Validación real en conjunto de test")
    print("🏆 Evaluación del modelo AUC 0.8592")
    print("🚀 Preparación para challenge submission")
    print("="*70)
    
    # Verificar que existen los archivos necesarios
    required_files = [
        'best_optimized_temporal_cnn.pth',
        # 'best_ensemble_model.pkl' # Opcional
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ ARCHIVOS FALTANTES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Asegúrate de tener los modelos entrenados")
        print("   Ejecuta primero el entrenamiento completo")
        return
    
    # Inicializar inferencia
    inference = MAMAMIATestInference(
        data_dir=data_dir,
        splits_csv=splits_csv,
        pcr_labels_file=pcr_labels_file
    )
    
    # Ejecutar inferencia
    results = inference.run_test_inference()
    
    # Resumen final
    if results:
        test_auc = results['test_auc']
        gap = results['generalization_gap']
        
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"🏆 AUC Test: {test_auc:.4f}")
        print(f"📊 Gap Train-Test: {gap:.4f}")
        
        if test_auc >= 0.85 and gap < 0.10:
            print("\n🎉 ¡MODELO EXCELENTE VALIDADO!")
            print("🏆 Listo para ganar el challenge")
        elif test_auc >= 0.80:
            print("\n🎊 ¡MODELO MUY BUENO VALIDADO!")
            print("💪 Competitivo para el challenge")
        else:
            print("\n📈 Modelo validado con margen de mejora")
        
        print(f"\n✅ Inferencia completada exitosamente")

if __name__ == "__main__":
    main()


# ==============================================================================
# INSTRUCCIONES DE USO
# ==============================================================================

"""
🎯 MAMA-MIA TEST INFERENCE - INSTRUCCIONES

1. REQUISITOS PREVIOS:
   ✅ Modelo temporal CNN entrenado: 'best_optimized_temporal_cnn.pth'
   ⚠️ Ensemble model (opcional): 'best_ensemble_model.pkl'
   ⚠️ Feature scaler (opcional): 'feature_scaler.pkl'

2. EJECUCIÓN:
   python mama_mia_test_inference.py

3. ARCHIVOS GENERADOS:
   📊 mama_mia_test_results.csv - Resultados completos con true labels
   🏆 mama_mia_challenge_submission.csv - Para submission al challenge
   📈 mama_mia_test_inference_results.json - Métricas detalladas

4. INTERPRETACIÓN RESULTADOS:
   🏆 AUC > 0.85: EXCELENTE, listo para challenge
   ✅ AUC > 0.80: MUY BUENO, competitivo
   📈 AUC > 0.75: BUENO, considerar submission
   ⚠️ AUC < 0.75: Necesita optimización

5. GENERALIZATION GAP:
   ✅ Gap < 0.05: Excelente generalización
   ✅ Gap < 0.10: Buena generalización  
   ⚠️ Gap > 0.15: Posible overfitting

6. SUBMISSION AL CHALLENGE:
   - Usar archivo: mama_mia_challenge_submission.csv
   - Formato: patient_id, pcr_probability
   - Solo si AUC test > 0.80 recomendado

OBJETIVO: Validar que el AUC 0.8592 de train se mantiene en test
"""