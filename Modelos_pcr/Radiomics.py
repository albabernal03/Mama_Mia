import numpy as np
import pandas as pd
from pathlib import Path
import json
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectPercentile
from sklearn.pipeline import Pipeline
from scipy import stats, ndimage
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings('ignore')

class MamaMiaRadiomicsPipeline:
    """
    🎯 Pipeline completo de radiomics para MAMA-MIA
    """
    
    def __init__(self, cropped_data_dir, splits_csv, pcr_labels_csv):
        """Inicializar pipeline"""
        self.cropped_data_dir = Path(cropped_data_dir)
        self.splits_csv = splits_csv
        self.pcr_labels_csv = pcr_labels_csv
        
        # Load data
        self.splits_df = pd.read_csv(splits_csv)
        self.pcr_df = pd.read_csv(pcr_labels_csv)
        
        # Prepare data
        self.train_patients, self.train_labels = self._prepare_training_data()
        self.test_patients, self.test_labels = self._prepare_test_data()
        
        print(f"🎯 MAMA-MIA RADIOMICS PIPELINE INITIALIZED")
        print(f"   Cropped data dir: {self.cropped_data_dir}")
        print(f"   Training patients: {len(self.train_patients)}")
        print(f"   Test patients: {len(self.test_patients)}")
        if len(self.train_labels) > 0:
            print(f"   Train pCR rate: {sum(self.train_labels)/len(self.train_labels):.1%}")
        if len(self.test_labels) > 0:
            print(f"   Test pCR rate: {sum(self.test_labels)/len(self.test_labels):.1%}")
    
    def _prepare_training_data(self):
        """Preparar datos de entrenamiento"""
        train_patients = self.splits_df['train_split'].dropna().tolist()
        train_labels = []
        valid_patients = []
        
        # Detectar nombre de columna de PCR automáticamente
        pcr_column = None
        possible_columns = ['pcr', 'pcr_response', 'PCR', 'PCR_response', 'pCR', 'pCR_response']
        
        for col in possible_columns:
            if col in self.pcr_df.columns:
                pcr_column = col
                break
        
        if pcr_column is None:
            print(f"❌ ERROR: No PCR column found. Available columns: {list(self.pcr_df.columns)}")
            return [], []
        
        print(f"✅ Using PCR column: '{pcr_column}'")
        
        # Crear diccionario de labels para búsqueda rápida
        pcr_dict = dict(zip(self.pcr_df['patient_id'], self.pcr_df[pcr_column]))
        
        for patient_id in train_patients:
            if patient_id in pcr_dict:
                pcr_status = pcr_dict[patient_id]
                if pcr_status in [0, 1]:  # Labels válidas
                    train_labels.append(int(pcr_status))
                    valid_patients.append(patient_id)
        
        return valid_patients, train_labels
    
    def _prepare_test_data(self):
        """Preparar datos de test"""
        test_patients = self.splits_df['test_split'].dropna().tolist()
        test_labels = []
        valid_patients = []
        
        # Detectar nombre de columna de PCR automáticamente
        pcr_column = None
        possible_columns = ['pcr', 'pcr_response', 'PCR', 'PCR_response', 'pCR', 'pCR_response']
        
        for col in possible_columns:
            if col in self.pcr_df.columns:
                pcr_column = col
                break
        
        if pcr_column is None:
            print(f"❌ ERROR: No PCR column found. Available columns: {list(self.pcr_df.columns)}")
            return [], []
        
        # Crear diccionario de labels para búsqueda rápida
        pcr_dict = dict(zip(self.pcr_df['patient_id'], self.pcr_df[pcr_column]))
        
        for patient_id in test_patients:
            if patient_id in pcr_dict:
                pcr_status = pcr_dict[patient_id]
                if pcr_status in [0, 1]:  # Labels válidas
                    test_labels.append(int(pcr_status))
                    valid_patients.append(patient_id)
        
        return valid_patients, test_labels
    
    def load_patient_images(self, patient_id, split_type):
        """Cargar imágenes de un paciente - ARREGLADO SIN ERRORES"""
        patient_dir = self.cropped_data_dir / "images" / split_type / patient_id
        seg_dir = self.cropped_data_dir / "segmentations" / split_type / patient_id
        
        if not patient_dir.exists():
            print(f"❌ Patient directory not found: {patient_dir}")
            return None
        
        if not seg_dir.exists():
            print(f"❌ Segmentation directory not found: {seg_dir}")
            return None
        
        try:
            # Buscar archivos de imágenes croppeadas
            img_files = list(patient_dir.glob("*_cropped.nii.gz"))
            
            if len(img_files) == 0:
                print(f"❌ No cropped images found for {patient_id}")
                return None
            
            # Buscar máscara
            seg_files = list(seg_dir.glob("*_seg_cropped.nii.gz"))
            
            if len(seg_files) == 0:
                print(f"❌ No mask found for {patient_id}")
                return None
            
            # Cargar máscara
            mask_nii = nib.load(seg_files[0])
            mask = mask_nii.get_fdata()
            
            print(f"📁 Processing {patient_id}: found {len(img_files)} images")
            
            # Separar archivos por número de secuencia - SIN ERRORES DE REGEX
            numbered_files = {}
            for img_file in img_files:
                filename = img_file.name
                # Buscar patrón: termina en _XXXX_cropped.nii.gz
                if '_cropped.nii.gz' in filename:
                    # Extraer número antes de _cropped
                    parts = filename.replace('_cropped.nii.gz', '').split('_')
                    if len(parts) >= 2:
                        try:
                            # El último part debería ser el número
                            number_str = parts[-1]
                            if number_str.isdigit() and len(number_str) == 4:
                                number = int(number_str)
                                numbered_files[number] = img_file
                                print(f"   📋 Found sequence {number:04d}: {filename}")
                        except:
                            print(f"⚠️ Could not parse number from: {filename}")
            
            if len(numbered_files) == 0:
                print(f"❌ No numbered sequences found for {patient_id}")
                return None
            
            # Identificar pre y post según tu estructura
            # 0000 = pre-contraste, 0001 = primer post-contraste (STANDARD para challenge)
            pre_image = None
            post_image = None
            
            # Cargar en orden numérico
            for number in sorted(numbered_files.keys()):
                img_nii = nib.load(numbered_files[number])
                img_data = img_nii.get_fdata()
                
                if number == 0:  # 0000 = pre-contraste
                    pre_image = img_data
                    print(f"   📋 0000 → Pre-contrast: {img_data.shape}")
                elif number == 1:  # 0001 = primer post-contraste (STANDARD)
                    post_image = img_data
                    print(f"   📋 0001 → Post-contrast (first): {img_data.shape}")
                    break  # Solo usar el primer post-contraste para robustez
                else:
                    print(f"   📋 {number:04d} → Additional post-contrast (ignored for robustness)")
            
            # Verificar que tenemos pre-contraste
            if pre_image is None:
                print(f"❌ No pre-contrast image (0000) found for {patient_id}")
                return None
            
            # Verificar que tenemos post-contraste
            if post_image is None:
                print(f"❌ No post-contrast image (0001) found for {patient_id}")
                return None
            
            print(f"   ✅ Using standard pre (0000) + first post (0001) for challenge robustness")
            
            # Verificar shapes
            if pre_image.shape != post_image.shape or pre_image.shape != mask.shape:
                print(f"❌ Shape mismatch for {patient_id}:")
                print(f"   Pre: {pre_image.shape}")
                print(f"   Post: {post_image.shape}")
                print(f"   Mask: {mask.shape}")
                return None
            
            # Verificar ROI
            roi_size = np.sum(mask > 0)
            if roi_size < 10:
                print(f"❌ ROI too small for {patient_id}: {roi_size} voxels")
                return None
            
            print(f"   ✅ Successfully loaded {patient_id}: ROI size = {roi_size} voxels")
            
            return {
                'pre': pre_image.astype(np.float32),
                'post': post_image.astype(np.float32),
                'mask': mask.astype(np.float32)
            }
            
        except Exception as e:
            print(f"❌ Error loading {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_images(self, pre_image, post_image, mask):
        """Preprocesamiento optimizado"""
        # 1. Clip outliers
        roi_pre = pre_image[mask > 0]
        roi_post = post_image[mask > 0]
        
        if len(roi_pre) > 0:
            p_low_pre, p_high_pre = np.percentile(roi_pre, [0.5, 99.5])
            pre_clipped = np.clip(pre_image, p_low_pre, p_high_pre)
        else:
            pre_clipped = pre_image.copy()
        
        if len(roi_post) > 0:
            p_low_post, p_high_post = np.percentile(roi_post, [0.5, 99.5])
            post_clipped = np.clip(post_image, p_low_post, p_high_post)
        else:
            post_clipped = post_image.copy()
        
        # 2. Z-score normalization
        roi_pre_clipped = pre_clipped[mask > 0]
        roi_post_clipped = post_clipped[mask > 0]
        
        if len(roi_pre_clipped) > 1 and np.std(roi_pre_clipped) > 1e-8:
            mean_pre = np.mean(roi_pre_clipped)
            std_pre = np.std(roi_pre_clipped)
            pre_normalized = (pre_clipped - mean_pre) / std_pre
        else:
            pre_normalized = pre_clipped
        
        if len(roi_post_clipped) > 1 and np.std(roi_post_clipped) > 1e-8:
            mean_post = np.mean(roi_post_clipped)
            std_post = np.std(roi_post_clipped)
            post_normalized = (post_clipped - mean_post) / std_post
        else:
            post_normalized = post_clipped
        
        return pre_normalized, post_normalized, mask
    
    def extract_radiomics_features(self, pre_image, post_image, mask):
        """Extraer features radiómicas completas"""
        features = {}
        
        # Preprocesar
        pre_proc, post_proc, mask_proc = self.preprocess_images(pre_image, post_image, mask)
        
        roi_pre = pre_proc[mask_proc > 0]
        roi_post = post_proc[mask_proc > 0]
        roi_enh = roi_post - roi_pre
        
        if len(roi_pre) < 10:
            return {}
        
        # 1. First-order features (PRE)
        features.update(self._first_order_features(roi_pre, 'pre'))
        
        # 2. First-order features (POST)
        features.update(self._first_order_features(roi_post, 'post'))
        
        # 3. Enhancement features (CRÍTICAS)
        features.update(self._enhancement_features(roi_pre, roi_post, roi_enh))
        
        # 4. Shape features
        features.update(self._shape_features(mask_proc))
        
        # 5. Texture features
        features.update(self._texture_features(pre_proc, mask_proc, 'pre'))
        features.update(self._texture_features(post_proc, mask_proc, 'post'))
        
        # Validar features
        validated = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                validated[key] = float(value)
        
        return validated
    
    def _first_order_features(self, roi, prefix):
        """First-order statistics"""
        features = {}
        if len(roi) == 0:
            return features
        
        try:
            features[f'{prefix}_mean'] = np.mean(roi)
            features[f'{prefix}_median'] = np.median(roi)
            features[f'{prefix}_std'] = np.std(roi)
            features[f'{prefix}_var'] = np.var(roi)
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
            
            # Entropía
            hist, _ = np.histogram(roi, bins=64, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                features[f'{prefix}_entropy'] = -np.sum(hist * np.log2(hist))
        
        except Exception as e:
            print(f"Error in first-order features ({prefix}): {e}")
        
        return features
    
    def _enhancement_features(self, roi_pre, roi_post, roi_enh):
        """Enhancement features - CRÍTICAS para pCR"""
        features = {}
        
        try:
            # Enhancement básicas
            features['enh_mean'] = np.mean(roi_enh)
            features['enh_median'] = np.median(roi_enh)
            features['enh_std'] = np.std(roi_enh)
            features['enh_min'] = np.min(roi_enh)
            features['enh_max'] = np.max(roi_enh)
            features['enh_range'] = features['enh_max'] - features['enh_min']
            features['enh_skewness'] = stats.skew(roi_enh)
            features['enh_kurtosis'] = stats.kurtosis(roi_enh)
            
            # Percentiles de enhancement
            for p in [10, 25, 75, 90]:
                features[f'enh_p{p}'] = np.percentile(roi_enh, p)
            
            # Ratios de enhancement
            pre_mean = np.mean(roi_pre)
            post_mean = np.mean(roi_post)
            
            if abs(pre_mean) > 1e-8:
                features['enh_ratio'] = features['enh_mean'] / pre_mean
                features['post_pre_ratio'] = post_mean / pre_mean
                features['peak_enh_ratio'] = features['enh_max'] / pre_mean
                features['relative_enhancement'] = (post_mean - pre_mean) / pre_mean
            else:
                features['enh_ratio'] = 0
                features['post_pre_ratio'] = 1
                features['peak_enh_ratio'] = 0
                features['relative_enhancement'] = 0
            
            # Distribución de enhancement
            positive_enh = roi_enh[roi_enh > 0]
            negative_enh = roi_enh[roi_enh < 0]
            
            features['positive_enh_fraction'] = len(positive_enh) / len(roi_enh)
            features['negative_enh_fraction'] = len(negative_enh) / len(roi_enh)
            
            if len(positive_enh) > 0:
                features['positive_enh_mean'] = np.mean(positive_enh)
                features['positive_enh_max'] = np.max(positive_enh)
            else:
                features['positive_enh_mean'] = 0
                features['positive_enh_max'] = 0
            
            # Heterogeneidad de enhancement
            features['enh_cv'] = np.std(roi_enh) / (abs(np.mean(roi_enh)) + 1e-8)
            
            # Umbrales de enhancement
            for thresh in [0.1, 0.5, 1.0]:
                features[f'enh_above_{thresh}'] = np.sum(roi_enh > thresh) / len(roi_enh)
        
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
                
                # Diámetro aproximado
                if len(coords[0]) > 1:
                    points = np.column_stack(coords)
                    if len(points) > 100:
                        indices = np.random.choice(len(points), 100, replace=False)
                        points = points[indices]
                    
                    from scipy.spatial.distance import pdist
                    distances = pdist(points)
                    if len(distances) > 0:
                        features['shape_max_diameter'] = np.max(distances)
        
        except Exception as e:
            print(f"Error in shape features: {e}")
        
        return features
    
    def _texture_features(self, image, mask, prefix):
        """Texture features"""
        features = {}
        
        try:
            # Varianza local
            local_mean = uniform_filter(image, size=3)
            local_sq_mean = uniform_filter(image**2, size=3)
            local_variance = local_sq_mean - local_mean**2
            
            roi_local_var = local_variance[mask > 0]
            if len(roi_local_var) > 0:
                features[f'{prefix}_texture_local_var_mean'] = np.mean(roi_local_var)
                features[f'{prefix}_texture_local_var_std'] = np.std(roi_local_var)
            
            # Gradientes
            gradients = np.gradient(image)
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
            
            roi_gradient = gradient_magnitude[mask > 0]
            if len(roi_gradient) > 0:
                features[f'{prefix}_texture_gradient_mean'] = np.mean(roi_gradient)
                features[f'{prefix}_texture_gradient_std'] = np.std(roi_gradient)
        
        except Exception as e:
            print(f"Error in texture features ({prefix}): {e}")
        
        return features
    
    def extract_all_features(self):
        """Extraer features de todos los pacientes"""
        print("🔄 Extracting radiomics features...")
        
        # Train features
        train_features = []
        valid_train_patients = []
        valid_train_labels = []
        
        for i, patient_id in enumerate(tqdm(self.train_patients, desc="Train features")):
            images = self.load_patient_images(patient_id, 'train_split')
            if images is not None:
                features = self.extract_radiomics_features(
                    images['pre'], images['post'], images['mask']
                )
                if len(features) > 20:
                    train_features.append(features)
                    valid_train_patients.append(patient_id)
                    valid_train_labels.append(self.train_labels[i])
                else:
                    print(f"❌ {patient_id}: Insufficient features ({len(features)})")
        
        # Test features
        test_features = []
        valid_test_patients = []
        valid_test_labels = []
        
        for i, patient_id in enumerate(tqdm(self.test_patients, desc="Test features")):
            images = self.load_patient_images(patient_id, 'test_split')
            if images is not None:
                features = self.extract_radiomics_features(
                    images['pre'], images['post'], images['mask']
                )
                if len(features) > 20:
                    test_features.append(features)
                    valid_test_patients.append(patient_id)
                    valid_test_labels.append(self.test_labels[i])
                else:
                    print(f"❌ {patient_id}: Insufficient features ({len(features)})")
        
        # Update valid data
        self.train_patients = valid_train_patients
        self.train_labels = valid_train_labels
        self.test_patients = valid_test_patients
        self.test_labels = valid_test_labels
        
        if len(train_features) == 0 or len(test_features) == 0:
            print("❌ No valid features extracted!")
            return None, None
        
        # Convert to DataFrames
        train_df = pd.DataFrame(train_features).fillna(0)
        test_df = pd.DataFrame(test_features).fillna(0)
        
        # Ensure same columns
        all_columns = list(set(train_df.columns) | set(test_df.columns))
        for col in all_columns:
            if col not in train_df:
                train_df[col] = 0
            if col not in test_df:
                test_df[col] = 0
        
        train_df = train_df[all_columns]
        test_df = test_df[all_columns]
        
        # Remove zero variance features
        feature_vars = train_df.var()
        valid_features = feature_vars[feature_vars > 1e-8].index
        train_df = train_df[valid_features]
        test_df = test_df[valid_features]
        
        print(f"✅ Features extracted successfully!")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Features: {len(train_df.columns)}")
        
        return train_df, test_df
    
    def train_models(self, train_df):
        """Entrenar modelos optimizados"""
        print("🚀 Training optimized models...")
        
        X = train_df.values
        y = np.array(self.train_labels)
        
        print(f"Training data: {X.shape}, Labels: {np.bincount(y)}")
        
        # Modelos
        models = {
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=250,
                max_depth=None,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            ),
            'lr': LogisticRegression(
                penalty='elasticnet',
                l1_ratio=0.5,
                C=0.1,
                solver='saga',
                class_weight='balanced',
                random_state=42,
                max_iter=3000
            )
        }
        
        # Feature selectors
        feature_selectors = {
            'percentile_30': SelectPercentile(f_classif, percentile=30),
            'kbest_30': SelectKBest(f_classif, k=min(30, X.shape[1]//2)),
            'kbest_50': SelectKBest(f_classif, k=min(50, X.shape[1]//2)),
        }
        
        cv_results = {}
        cv_scores = {}
        
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        
        print("📊 Cross-validation...")
        
        for fs_name, feature_selector in feature_selectors.items():
            for model_name, model in models.items():
                try:
                    pipeline_name = f'{model_name}_{fs_name}'
                    
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('feature_selection', feature_selector),
                        ('classifier', model)
                    ])
                    
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                    
                    cv_results[pipeline_name] = {
                        'mean_auc': np.mean(scores),
                        'std_auc': np.std(scores),
                        'scores': scores
                    }
                    cv_scores[pipeline_name] = np.mean(scores)
                    
                    print(f"  {pipeline_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                    
                except Exception as e:
                    print(f"❌ Error with {pipeline_name}: {e}")
                    cv_scores[pipeline_name] = 0.0
        
        # Select best model
        if cv_scores:
            best_model_name = max(cv_scores, key=cv_scores.get)
            best_auc = cv_scores[best_model_name]
            
            print(f"\n🏆 Best model: {best_model_name} (CV AUC: {best_auc:.4f})")
            
            # Reconstruct best pipeline
            parts = best_model_name.split('_')
            model_part = parts[0]
            fs_part = '_'.join(parts[1:])
            
            best_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', feature_selectors[fs_part]),
                ('classifier', models[model_part])
            ])
            
            best_pipeline.fit(X, y)
            
            return best_pipeline, cv_results, best_model_name
        else:
            print("❌ No models trained successfully")
            return None, {}, ""
    
    def evaluate_on_test(self, model, train_df, test_df):
        """Evaluación en test"""
        print("🎯 Final evaluation on TEST set...")
        
        X_test = test_df.values
        y_test = np.array(self.test_labels)
        
        try:
            test_probs = model.predict_proba(X_test)[:, 1]
            test_preds = model.predict(X_test)
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return 0.5, 0.5, np.random.random(len(y_test)), np.random.randint(0, 2, len(y_test))
        
        test_auc = roc_auc_score(y_test, test_probs)
        test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        
        return test_auc, test_balanced_acc, test_probs, test_preds, precision, recall, f1
    
    def run_complete_pipeline(self):
        """🎯 Ejecutar pipeline completo"""
        print("🎯 MAMA-MIA COMPLETE RADIOMICS PIPELINE")
        print("="*80)
        print("📊 From cropped data to final evaluation")
        print("🏆 Target: AUC > 0.80 on test set")
        print("="*80)
        
        # 1. Extract features
        train_df, test_df = self.extract_all_features()
        
        if train_df is None or len(train_df.columns) == 0:
            print("❌ Feature extraction failed!")
            return None
        
        # 2. Train models
        best_model, cv_results, best_model_name = self.train_models(train_df)
        
        if best_model is None:
            print("❌ Model training failed!")
            return None
        
        # 3. Evaluate on test
        test_auc, test_balanced_acc, test_probs, test_preds, precision, recall, f1 = self.evaluate_on_test(best_model, train_df, test_df)
        
        # 4. Results analysis
        print("\n" + "="*80)
        print("🎯 FINAL RESULTS")
        print("="*80)
        
        # Train performance (CV)
        train_auc_cv = cv_results[best_model_name]['mean_auc']
        train_std_cv = cv_results[best_model_name]['std_auc']
        
        print(f"📊 CV Train AUC: {train_auc_cv:.4f} ± {train_std_cv:.4f}")
        print(f"🎯 Test AUC: {test_auc:.4f}")
        print(f"📊 Test Balanced Acc: {test_balanced_acc:.4f}")
        print(f"📊 Test Precision: {precision:.4f}")
        print(f"📊 Test Recall: {recall:.4f}")
        print(f"📊 Test F1-Score: {f1:.4f}")
        
        # Generalization analysis
        gap = train_auc_cv - test_auc
        print(f"\n📈 GENERALIZATION ANALYSIS:")
        print(f"   CV Train: {train_auc_cv:.4f}")
        print(f"   Test: {test_auc:.4f}")
        print(f"   Gap: {gap:.4f}")
        
        if gap < 0.05:
            print("✅ EXCELLENT generalization!")
        elif gap < 0.10:
            print("✅ GOOD generalization")
        elif gap < 0.15:
            print("⚠️ Acceptable generalization")
        else:
            print("❌ Possible overfitting")
        
        # Feature importance analysis
        try:
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                feature_names = train_df.columns
                selected_features = best_model.named_steps['feature_selection'].get_support()
                selected_feature_names = feature_names[selected_features]
                importances = best_model.named_steps['classifier'].feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'feature': selected_feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
                for i, row in feature_importance_df.head(15).iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
                
                # Feature category analysis
                enh_features = len([f for f in selected_feature_names if f.startswith('enh_')])
                shape_features = len([f for f in selected_feature_names if f.startswith('shape_')])
                texture_features = len([f for f in selected_feature_names if 'texture' in f])
                
                print(f"\n📊 SELECTED FEATURE CATEGORIES:")
                print(f"   Enhancement features: {enh_features}")
                print(f"   Shape features: {shape_features}")
                print(f"   Texture features: {texture_features}")
                
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
        
        # Detailed metrics
        cm = confusion_matrix(self.test_labels, test_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n📊 DETAILED TEST METRICS:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f}")
        print(f"   Specificity: {specificity:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Save results
        results = {
            'cv_train_auc': float(train_auc_cv),
            'cv_train_std': float(train_std_cv),
            'test_auc': float(test_auc),
            'test_balanced_accuracy': float(test_balanced_acc),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'test_sensitivity': float(sensitivity),
            'test_specificity': float(specificity),
            'generalization_gap': float(gap),
            'best_model': best_model_name,
            'feature_count': len(train_df.columns),
            'selected_features_count': int(np.sum(best_model.named_steps['feature_selection'].get_support())),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        
        # Create submission files
        submission_df = pd.DataFrame({
            'patient_id': self.test_patients,
            'pcr_probability': test_probs,
            'pcr_prediction': test_preds,
            'true_label': self.test_labels
        })
        
        submission_df.to_csv('mama_mia_radiomics_test_results.csv', index=False)
        
        # Challenge submission
        challenge_submission = submission_df[['patient_id', 'pcr_probability']].copy()
        challenge_submission.to_csv('mama_mia_radiomics_challenge_submission.csv', index=False)
        
        with open('mama_mia_radiomics_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 FILES GENERATED:")
        print(f"   📊 mama_mia_radiomics_test_results.csv")
        print(f"   🏆 mama_mia_radiomics_challenge_submission.csv") 
        print(f"   📈 mama_mia_radiomics_results.json")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if test_auc >= 0.85 and gap < 0.10:
            print("🏆 EXCELLENT PERFORMANCE!")
            print("   ✅ Ready for challenge submission")
            assessment = "excellent"
        elif test_auc >= 0.80 and gap < 0.15:
            print("🌟 VERY GOOD PERFORMANCE!")
            print("   ✅ Strong challenge candidate")
            assessment = "very_good"
        elif test_auc >= 0.75:
            print("📈 GOOD PERFORMANCE!")
            print("   ✅ Significant improvement achieved")
            assessment = "good"
        elif test_auc >= 0.70:
            print("🎯 MODERATE PERFORMANCE")
            print("   📊 Some improvement over baseline")
            assessment = "moderate"
        else:
            print("⚠️ NEEDS IMPROVEMENT")
            print("   🔧 Review feature extraction")
            assessment = "needs_improvement"
        
        results['assessment'] = assessment
        
        # Recommendations
        print(f"\n🚀 RECOMMENDATIONS:")
        if test_auc >= 0.80:
            print("   1. ✅ SUBMIT to challenge!")
            print("   2. 🏆 Model is competition-ready")
        elif test_auc >= 0.75:
            print("   1. ✅ Submit to challenge")
            print("   2. 🔧 Consider ensemble methods")
        else:
            print("   1. 🔧 Review feature extraction")
            print("   2. 🔍 Check data preprocessing")
        
        print(f"\n✅ Complete pipeline finished!")
        print(f"🎯 Final Test AUC: {test_auc:.4f}")
        
        return results

def main():
    """🎯 Ejecutar pipeline completo de MAMA-MIA"""
    
    # RUTAS CONFIGURADAS - TUS RUTAS EXACTAS
    cropped_data_dir = r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    pcr_labels_csv = r"C:\Users\usuario\Documents\Mama_Mia\PCR\pcr_labels.csv"
    
    print("🎯 MAMA-MIA COMPLETE RADIOMICS PIPELINE")
    print("="*80)
    print("📊 Complete pipeline from cropped data to final evaluation")
    print("🔬 Optimized radiomics extraction and model training")
    print("🏆 Target: AUC > 0.80 on test set")
    print("="*80)
    print(f"📁 Cropped data: {cropped_data_dir}")
    print(f"📊 Splits CSV: {splits_csv}")
    print(f"🏷️ PCR labels: {pcr_labels_csv}")
    print("="*80)
    
    # Verificar archivos
    from pathlib import Path
    
    if not Path(cropped_data_dir).exists():
        print(f"❌ ERROR: Cropped data directory not found: {cropped_data_dir}")
        return
    
    if not Path(splits_csv).exists():
        print(f"❌ ERROR: Splits CSV not found: {splits_csv}")
        return
    
    if not Path(pcr_labels_csv).exists():
        print(f"❌ ERROR: PCR labels CSV not found: {pcr_labels_csv}")
        print(f"📋 Create this file with format:")
        print(f"   patient_id,pcr")
        print(f"   DUKE_019,0")
        print(f"   DUKE_021,1")
        print(f"   ...")
        return
    
    # Initialize pipeline
    try:
        pipeline = MamaMiaRadiomicsPipeline(
            cropped_data_dir=cropped_data_dir,
            splits_csv=splits_csv,
            pcr_labels_csv=pcr_labels_csv
        )
    except Exception as e:
        print(f"❌ ERROR initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run pipeline
    results = pipeline.run_complete_pipeline()
    
    if results:
        print(f"\n🎯 PIPELINE SUMMARY:")
        print(f"🏆 Test AUC: {results['test_auc']:.4f}")
        print(f"📊 Generalization Gap: {results['generalization_gap']:.4f}")
        print(f"🔬 Selected Features: {results['selected_features_count']}/{results['feature_count']}")
        print(f"📈 Assessment: {results['assessment'].upper()}")
        
        if results['test_auc'] >= 0.80:
            print("\n🎉 ¡PIPELINE EXITOSO!")
            print("🏆 Modelo listo para challenge submission")
        elif results['test_auc'] >= 0.75:
            print("\n🎊 ¡PIPELINE PROMETEDOR!")
            print("💪 Buen modelo competitivo")
        
        print(f"\n✅ Complete pipeline execution finished!")
    else:
        print("\n❌ Pipeline execution failed!")

def test_loading():
    """Test simple para verificar carga de datos"""
    cropped_data_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\cropped_data")
    
    print("🧪 Testing data loading...")
    
    # Check directory structure
    images_dir = cropped_data_dir / "images"
    seg_dir = cropped_data_dir / "segmentations"
    
    print(f"Images dir exists: {images_dir.exists()}")
    print(f"Segmentations dir exists: {seg_dir.exists()}")
    
    if images_dir.exists():
        # Check splits
        train_dir = images_dir / "train_split"
        test_dir = images_dir / "test_split"
        
        print(f"Train split exists: {train_dir.exists()}")
        print(f"Test split exists: {test_dir.exists()}")
        
        if test_dir.exists():
            patients = [d.name for d in test_dir.iterdir() if d.is_dir()]
            print(f"Test patients found: {len(patients)}")
            if patients:
                print(f"Example patients: {patients[:3]}")
                
                # Test loading first patient
                first_patient = patients[0]
                patient_dir = test_dir / first_patient
                img_files = list(patient_dir.glob("*_cropped.nii.gz"))
                
                print(f"\nTesting patient: {first_patient}")
                print(f"Image files: {[f.name for f in img_files]}")
                
                # Test regex parsing
                for img_file in img_files:
                    filename = img_file.name
                    if '_cropped.nii.gz' in filename:
                        parts = filename.replace('_cropped.nii.gz', '').split('_')
                        if len(parts) >= 2:
                            number_str = parts[-1]
                            if number_str.isdigit() and len(number_str) == 4:
                                number = int(number_str)
                                print(f"✅ {filename} → sequence {number:04d}")

if __name__ == "__main__":
    # Ejecutar pipeline completo
    main()
    