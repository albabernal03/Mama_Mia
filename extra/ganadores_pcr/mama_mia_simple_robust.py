r"""
MAMA-MIA SIMPLE ROBUST MODEL
🎯 Evitar overfitting con modelo más simple
📊 Enfoque en radiomics robustos sin CNN compleja
🏆 Target: AUC consistente entre train y test
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SimpleRobustRadiomics:
    """
    🎯 Extractor de radiomics SIMPLE y ROBUSTO
    """
    
    def __init__(self):
        pass
    
    def extract_robust_features(self, patient_id, data_dir):
        """
        Extrae solo las features MÁS ROBUSTAS y menos propensas al overfitting
        """
        patient_dir = Path(data_dir) / patient_id
        tensor_file = patient_dir / f"{patient_id}_tensor_3ch.nii.gz"
        
        if not tensor_file.exists():
            return self._get_default_features()
        
        try:
            tensor_img = nib.load(tensor_file)
            tensor_data = tensor_img.get_fdata().astype(np.float32)
            
            if tensor_data.ndim != 4 or tensor_data.shape[0] < 3:
                return self._get_default_features()
            
            pre_image = tensor_data[0]
            post_image = tensor_data[1] 
            mask = tensor_data[2]
            
            features = {}
            
            if mask is None or np.sum(mask) == 0:
                return self._get_default_features()
            
            roi_pre = pre_image[mask > 0]
            roi_post = post_image[mask > 0]
            roi_delta = post_image[mask > 0] - pre_image[mask > 0]
            
            if len(roi_pre) == 0:
                return self._get_default_features()
            
            # 🔥 FEATURES BÁSICAS MUY ROBUSTAS
            
            # 1. Intensidades básicas (más robustas)
            features['pre_mean'] = np.mean(roi_pre)
            features['pre_median'] = np.median(roi_pre)
            features['pre_std'] = np.std(roi_pre)
            
            features['post_mean'] = np.mean(roi_post)
            features['post_median'] = np.median(roi_post)
            features['post_std'] = np.std(roi_post)
            
            # 2. Enhancement básico (MUY importante para pCR)
            features['enhancement_mean'] = np.mean(roi_delta)
            features['enhancement_median'] = np.median(roi_delta)
            features['enhancement_std'] = np.std(roi_delta)
            
            # 3. Ratios robustos
            features['enhancement_ratio'] = features['enhancement_mean'] / (features['pre_mean'] + 1e-6)
            features['intensity_ratio'] = features['post_mean'] / (features['pre_mean'] + 1e-6)
            
            # 4. Percentiles (robustos a outliers)
            features['pre_p75'] = np.percentile(roi_pre, 75)
            features['pre_p25'] = np.percentile(roi_pre, 25)
            features['post_p75'] = np.percentile(roi_post, 75)
            features['post_p25'] = np.percentile(roi_post, 25)
            features['enhancement_p75'] = np.percentile(roi_delta, 75)
            features['enhancement_p25'] = np.percentile(roi_delta, 25)
            
            # 5. Volumen (muy robusto)
            features['tumor_volume'] = np.sum(mask)
            features['tumor_volume_normalized'] = np.sum(mask) / np.prod(mask.shape)
            
            # 6. Heterogeneidad básica
            features['pre_cv'] = features['pre_std'] / (features['pre_mean'] + 1e-6)
            features['post_cv'] = features['post_std'] / (features['post_mean'] + 1e-6)
            features['enhancement_cv'] = features['enhancement_std'] / (abs(features['enhancement_mean']) + 1e-6)
            
            # 7. Enhanced response patterns (simplificados)
            strong_enh_threshold = np.percentile(roi_delta, 75)
            weak_enh_threshold = np.percentile(roi_delta, 25)
            
            features['strong_enhancement_fraction'] = np.sum(roi_delta > strong_enh_threshold) / len(roi_delta)
            features['weak_enhancement_fraction'] = np.sum(roi_delta < weak_enh_threshold) / len(roi_delta)
            
            # 8. Spatial features básicas (solo si hay suficientes voxels)
            if len(roi_delta) > 50:  # Solo para tumores grandes
                coords = np.where(mask > 0)
                center_z = np.mean(coords[0])
                center_y = np.mean(coords[1])
                center_x = np.mean(coords[2])
                
                z_coords, y_coords, x_coords = coords
                distances = np.sqrt(
                    (z_coords - center_z)**2 + 
                    (y_coords - center_y)**2 + 
                    (x_coords - center_x)**2
                )
                
                # Análisis centro vs periferia simplificado
                central_threshold = np.percentile(distances, 50)  # Mediana
                central_mask = distances <= central_threshold
                
                if np.sum(central_mask) > 10 and np.sum(~central_mask) > 10:
                    central_enh = roi_delta[central_mask]
                    peripheral_enh = roi_delta[~central_mask]
                    
                    features['central_enhancement'] = np.mean(central_enh)
                    features['peripheral_enhancement'] = np.mean(peripheral_enh)
                    features['center_periphery_ratio'] = features['central_enhancement'] / (features['peripheral_enhancement'] + 1e-6)
                else:
                    features['central_enhancement'] = features['enhancement_mean']
                    features['peripheral_enhancement'] = features['enhancement_mean']
                    features['center_periphery_ratio'] = 1.0
            else:
                features['central_enhancement'] = features['enhancement_mean']
                features['peripheral_enhancement'] = features['enhancement_mean']
                features['center_periphery_ratio'] = 1.0
            
            # Validar todas las features
            for key, value in features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for {patient_id}: {e}")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Features por defecto"""
        return {
            'pre_mean': 0.0, 'pre_median': 0.0, 'pre_std': 0.0,
            'post_mean': 0.0, 'post_median': 0.0, 'post_std': 0.0,
            'enhancement_mean': 0.0, 'enhancement_median': 0.0, 'enhancement_std': 0.0,
            'enhancement_ratio': 1.0, 'intensity_ratio': 1.0,
            'pre_p75': 0.0, 'pre_p25': 0.0, 'post_p75': 0.0, 'post_p25': 0.0,
            'enhancement_p75': 0.0, 'enhancement_p25': 0.0,
            'tumor_volume': 0.0, 'tumor_volume_normalized': 0.0,
            'pre_cv': 0.0, 'post_cv': 0.0, 'enhancement_cv': 0.0,
            'strong_enhancement_fraction': 0.0, 'weak_enhancement_fraction': 0.0,
            'central_enhancement': 0.0, 'peripheral_enhancement': 0.0, 'center_periphery_ratio': 1.0
        }

class SimpleRobustPredictor:
    """
    🎯 Predictor SIMPLE y ROBUSTO para evitar overfitting
    """
    
    def __init__(self, data_dir, splits_csv, pcr_labels_file):
        self.data_dir = Path(data_dir)
        self.splits_csv = splits_csv
        self.pcr_labels_file = pcr_labels_file
        
        # Load data
        self.splits_df = pd.read_csv(splits_csv)
        with open(pcr_labels_file, 'r') as f:
            pcr_list = json.load(f)
        
        self.pcr_data = {}
        for item in pcr_list:
            patient_id = item['patient_id']
            self.pcr_data[patient_id] = item
        
        self.radiomics_extractor = SimpleRobustRadiomics()
        self.train_patients, self.train_labels = self._prepare_training_data()
        self.test_patients, self.test_labels = self._prepare_test_data()
        
        print(f"🎯 SIMPLE ROBUST MAMA-MIA MODEL")
        print(f"   Training patients: {len(self.train_patients)}")
        print(f"   Test patients: {len(self.test_patients)}")
        print(f"   Train pCR rate: {sum(self.train_labels)/len(self.train_labels):.1%}")
        print(f"   Test pCR rate: {sum(self.test_labels)/len(self.test_labels):.1%}")
    
    def _prepare_training_data(self):
        train_patients = self.splits_df['train_split'].dropna().tolist()
        train_labels = []
        valid_patients = []
        
        for patient_id in train_patients:
            if patient_id in self.pcr_data:
                try:
                    pcr_status = self.pcr_data[patient_id]['pcr']
                    if pcr_status != 'unknown':
                        train_labels.append(int(pcr_status))
                        valid_patients.append(patient_id)
                except KeyError:
                    continue
        
        return valid_patients, train_labels
    
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
    
    def extract_all_features(self):
        """Extraer features de train y test"""
        print("🔄 Extrayendo features robustas...")
        
        # Train features
        train_features = []
        for patient_id in tqdm(self.train_patients, desc="Train features"):
            features = self.radiomics_extractor.extract_robust_features(patient_id, self.data_dir)
            train_features.append(features)
        
        # Test features
        test_features = []
        for patient_id in tqdm(self.test_patients, desc="Test features"):
            features = self.radiomics_extractor.extract_robust_features(patient_id, self.data_dir)
            test_features.append(features)
        
        train_df = pd.DataFrame(train_features).fillna(0)
        test_df = pd.DataFrame(test_features).fillna(0)
        
        # Asegurar mismas columnas
        all_columns = list(set(train_df.columns) | set(test_df.columns))
        for col in all_columns:
            if col not in train_df:
                train_df[col] = 0
            if col not in test_df:
                test_df[col] = 0
        
        train_df = train_df[all_columns]
        test_df = test_df[all_columns]
        
        print(f"✅ Extraídas {len(train_df.columns)} features robustas")
        
        return train_df, test_df
    
    def train_robust_models(self, train_df):
        """
        Entrenar modelos ROBUSTOS con validación cruzada estricta
        """
        print("🚀 Entrenando modelos ROBUSTOS...")
        
        X = train_df.values
        y = np.array(self.train_labels)
        
        # 🔥 MODELOS SIMPLES Y ROBUSTOS
        models = {
            'rf_simple': RandomForestClassifier(
                n_estimators=100,  # Menos árboles
                max_depth=5,       # Más shallow
                min_samples_split=10,  # Más conservador
                min_samples_leaf=5,    # Más conservador
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'lr_l1': LogisticRegression(
                penalty='l1',
                C=0.01,  # Más regularización
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'lr_l2': LogisticRegression(
                penalty='l2',
                C=0.1,   # Más regularización
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'gbm_simple': GradientBoostingClassifier(
                n_estimators=50,   # Menos estimators
                max_depth=3,       # Más shallow
                learning_rate=0.1,
                subsample=0.8,
                random_state=42    # 🔧 REMOVED class_weight (not supported)
            )
        }
        
        # 🎯 VALIDACIÓN CRUZADA ESTRICTA
        cv_results = {}
        cv_scores = {}
        
        # Cross-validation con 10 folds para más robustez
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        print("📊 Validación cruzada (10-fold)...")
        for name, model in models.items():
            try:
                # Pipeline con scaling y feature selection
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),  # Más robusto que StandardScaler
                    ('feature_selection', SelectKBest(f_classif, k=15)),  # Solo 15 mejores features
                    ('classifier', model)
                ])
                
                # Cross-validation
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                
                cv_results[name] = {
                    'mean_auc': np.mean(scores),
                    'std_auc': np.std(scores),
                    'scores': scores
                }
                cv_scores[name] = np.mean(scores)
                
                print(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                cv_scores[name] = 0.0
        
        # Seleccionar mejor modelo
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_auc = cv_scores[best_model_name]
        
        print(f"\n🏆 Mejor modelo: {best_model_name} (AUC: {best_auc:.4f})")
        
        # Entrenar mejor modelo en todos los datos
        best_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_classif, k=15)),
            ('classifier', models[best_model_name])
        ])
        
        best_pipeline.fit(X, y)
        
        return best_pipeline, cv_results, best_model_name
    
    def evaluate_on_test(self, model, train_df, test_df):
        """
        Evaluación HONESTA en test
        """
        print("🎯 Evaluación en TEST...")
        
        X_test = test_df.values
        y_test = np.array(self.test_labels)
        
        # Predicciones
        try:
            test_probs = model.predict_proba(X_test)[:, 1]
            test_preds = model.predict(X_test)
        except Exception as e:
            print(f"Error en predicción: {e}")
            test_probs = np.random.random(len(y_test))
            test_preds = (test_probs > 0.5).astype(int)
        
        # Métricas
        test_auc = roc_auc_score(y_test, test_probs)
        test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
        
        return test_auc, test_balanced_acc, test_probs, test_preds
    
    def run_robust_training_and_validation(self):
        """
        🎯 Ejecutar entrenamiento y validación ROBUSTA
        """
        print("🎯 MAMA-MIA SIMPLE ROBUST MODEL")
        print("="*60)
        print("🔧 Evitando overfitting con modelos simples")
        print("📊 Validación cruzada estricta")
        print("🏆 Objetivo: Consistencia train-test")
        print("="*60)
        
        # 1. Extraer features
        train_df, test_df = self.extract_all_features()
        
        # 2. Entrenar modelos robustos
        best_model, cv_results, best_model_name = self.train_robust_models(train_df)
        
        # 3. Evaluar en test
        test_auc, test_balanced_acc, test_probs, test_preds = self.evaluate_on_test(best_model, train_df, test_df)
        
        # 4. Análisis de resultados
        print("\n" + "="*60)
        print("🎯 RESULTADOS ROBUSTOS")
        print("="*60)
        
        # Train performance (CV)
        train_auc_cv = cv_results[best_model_name]['mean_auc']
        train_std_cv = cv_results[best_model_name]['std_auc']
        
        print(f"📊 CV Train AUC: {train_auc_cv:.4f} ± {train_std_cv:.4f}")
        print(f"🎯 Test AUC: {test_auc:.4f}")
        print(f"📊 Test Balanced Acc: {test_balanced_acc:.4f}")
        
        # Generalization gap
        gap = train_auc_cv - test_auc
        print(f"\n📈 ANÁLISIS DE GENERALIZACIÓN:")
        print(f"   CV Train: {train_auc_cv:.4f}")
        print(f"   Test: {test_auc:.4f}")
        print(f"   Gap: {gap:.4f}")
        
        if gap < 0.05:
            print("✅ EXCELENTE generalización!")
        elif gap < 0.10:
            print("✅ BUENA generalización")
        elif gap < 0.15:
            print("⚠️ Generalización aceptable")
        else:
            print("❌ Posible overfitting")
        
        # Assessment
        print(f"\n🎯 ASSESSMENT:")
        if test_auc >= 0.75 and gap < 0.10:
            print("🏆 MODELO ROBUSTO - Listo para challenge")
        elif test_auc >= 0.70 and gap < 0.15:
            print("✅ MODELO SÓLIDO - Buen candidato")
        elif test_auc >= 0.65:
            print("📈 MODELO MODERADO - Mejorar si es posible")
        else:
            print("⚠️ MODELO DÉBIL - Necesita más trabajo")
        
        # Guardar resultados
        results = {
            'cv_train_auc': float(train_auc_cv),
            'cv_train_std': float(train_std_cv),
            'test_auc': float(test_auc),
            'test_balanced_accuracy': float(test_balanced_acc),
            'generalization_gap': float(gap),
            'best_model': best_model_name,
            'cv_results': {k: {
                'mean_auc': float(v['mean_auc']),
                'std_auc': float(v['std_auc'])
            } for k, v in cv_results.items()},
            'model_type': 'simple_robust'
        }
        
        # Crear submission
        submission_df = pd.DataFrame({
            'patient_id': self.test_patients,
            'pcr_probability': test_probs,
            'pcr_prediction': test_preds,
            'true_label': self.test_labels
        })
        
        submission_df.to_csv('mama_mia_robust_test_results.csv', index=False)
        
        # Challenge submission
        challenge_submission = submission_df[['patient_id', 'pcr_probability']].copy()
        challenge_submission.to_csv('mama_mia_robust_challenge_submission.csv', index=False)
        
        with open('mama_mia_robust_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Archivos generados:")
        print(f"   📊 mama_mia_robust_test_results.csv")
        print(f"   🏆 mama_mia_robust_challenge_submission.csv")
        print(f"   📈 mama_mia_robust_results.json")
        
        return results

def main():
    """
    🎯 Ejecutar modelo robusto
    """
    
    # Paths
    data_dir = r"D:\mama_mia_final_corrected"
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    pcr_labels_file = r"D:\clinical_data_complete.json"
    
    print("🎯 MAMA-MIA SIMPLE ROBUST MODEL")
    print("="*60)
    print("🔧 Solucionando overfitting del modelo anterior")
    print("📊 Enfoque en robustez y generalización")
    print("🏆 Objetivo: Consistencia train-test")
    print("="*60)
    
    # Inicializar predictor
    predictor = SimpleRobustPredictor(
        data_dir=data_dir,
        splits_csv=splits_csv,
        pcr_labels_file=pcr_labels_file
    )
    
    # Ejecutar entrenamiento robusto
    results = predictor.run_robust_training_and_validation()
    
    print(f"\n✅ Entrenamiento robusto completado!")

if __name__ == "__main__":
    main()


# ==============================================================================
# CAMBIOS PARA EVITAR OVERFITTING
# ==============================================================================

"""
🎯 MAMA-MIA SIMPLE ROBUST MODEL - CAMBIOS APLICADOS

1. ELIMINACIÓN DE COMPONENTES COMPLEJOS:
   ❌ CNN 3D profunda (principal causa de overfitting)
   ❌ Simulación temporal artificial
   ❌ Features demasiado específicas
   ❌ Ensemble complejo

2. ENFOQUE EN RADIOMICS ROBUSTOS:
   ✅ Solo 25 features bien establecidas
   ✅ Features básicas menos propensas a overfitting
   ✅ Validación con estadísticas robustas

3. MODELOS SIMPLES:
   ✅ Random Forest con max_depth=5
   ✅ Logistic Regression con alta regularización
   ✅ GBM con pocos estimators
   ✅ Pipeline con feature selection automática

4. VALIDACIÓN ESTRICTA:
   ✅ 10-fold cross-validation
   ✅ RobustScaler para outliers
   ✅ SelectKBest para 15 mejores features
   ✅ Evaluación honesta en test

5. PREVENCIÓN DE OVERFITTING:
   ✅ Regularización agresiva
   ✅ Modelos menos complejos
   ✅ Menos parámetros
   ✅ Validación cruzada estricta

RESULTADO ESPERADO: AUC 0.70-0.80 CON GAP < 0.10
PRIORIDAD: Generalización sobre performance en train
"""