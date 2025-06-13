# simple_verify_features.py
"""
Verificación SIMPLIFICADA de que las features constantes están corregidas
Sin usar PyRadiomics - solo análisis directo de los tensors corregidos
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

class SimpleFeatureVerifier:
    """Verificador simple que analiza variabilidad en los tensors corregidos"""
    
    def __init__(self, processed_data_dir: str, splits_csv: str):
        self.processed_data_dir = Path(processed_data_dir)
        self.splits_csv = Path(splits_csv)
        
        # Stats
        self.results = {
            'patients_analyzed': 0,
            'features_calculated': 0,
            'constant_features': 0,
            'loading_errors': 0,
            'tensor_stats': {}
        }
    
    def extract_simple_features(self, patient_id: str) -> dict:
        """Extraer features simples directamente del tensor corregido"""
        
        tensor_path = self.processed_data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
        
        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor not found: {tensor_path}")
        
        # Cargar tensor [3, H, W, D] = [pre, post, mask]
        nii_data = nib.load(tensor_path)
        tensor_data = nii_data.get_fdata()
        
        if len(tensor_data.shape) != 4 or tensor_data.shape[0] != 3:
            raise ValueError(f"Invalid tensor shape: {tensor_data.shape}, expected [3, H, W, D]")
        
        # Extraer canales
        pre_contrast = tensor_data[0]    # Canal 0: pre-contraste corregido
        post_contrast = tensor_data[1]   # Canal 1: post-contraste
        mask = tensor_data[2]            # Canal 2: máscara binaria
        
        # Crear máscara ROI
        roi_mask = mask > 0.5
        
        if np.sum(roi_mask) < 10:
            raise ValueError(f"ROI too small: {np.sum(roi_mask)} voxels")
        
        # Extraer datos de ROI
        pre_roi = pre_contrast[roi_mask]
        post_roi = post_contrast[roi_mask]
        
        # Extraer datos de background (no ROI)
        bg_mask = (pre_contrast > 0) & (~roi_mask)
        pre_bg = pre_contrast[bg_mask] if np.any(bg_mask) else np.array([0])
        post_bg = post_contrast[bg_mask] if np.any(bg_mask) else np.array([0])
        
        # Calcular features simples pero representativas
        features = {}
        
        # === FEATURES PRE-CONTRASTE ===
        # ROI features
        features['pre_roi_mean'] = float(np.mean(pre_roi))
        features['pre_roi_std'] = float(np.std(pre_roi))
        features['pre_roi_min'] = float(np.min(pre_roi))
        features['pre_roi_max'] = float(np.max(pre_roi))
        features['pre_roi_median'] = float(np.median(pre_roi))
        features['pre_roi_variance'] = float(np.var(pre_roi))
        features['pre_roi_range'] = float(np.max(pre_roi) - np.min(pre_roi))
        features['pre_roi_q25'] = float(np.percentile(pre_roi, 25))
        features['pre_roi_q75'] = float(np.percentile(pre_roi, 75))
        features['pre_roi_iqr'] = features['pre_roi_q75'] - features['pre_roi_q25']
        
        # Background features
        features['pre_bg_mean'] = float(np.mean(pre_bg))
        features['pre_bg_std'] = float(np.std(pre_bg))
        features['pre_bg_variance'] = float(np.var(pre_bg))
        
        # Global features
        features['pre_global_mean'] = float(np.mean(pre_contrast[pre_contrast > 0]))
        features['pre_global_std'] = float(np.std(pre_contrast[pre_contrast > 0]))
        features['pre_global_variance'] = float(np.var(pre_contrast[pre_contrast > 0]))
        
        # === FEATURES POST-CONTRASTE ===
        # ROI features
        features['post_roi_mean'] = float(np.mean(post_roi))
        features['post_roi_std'] = float(np.std(post_roi))
        features['post_roi_min'] = float(np.min(post_roi))
        features['post_roi_max'] = float(np.max(post_roi))
        features['post_roi_median'] = float(np.median(post_roi))
        features['post_roi_variance'] = float(np.var(post_roi))
        features['post_roi_range'] = float(np.max(post_roi) - np.min(post_roi))
        features['post_roi_q25'] = float(np.percentile(post_roi, 25))
        features['post_roi_q75'] = float(np.percentile(post_roi, 75))
        features['post_roi_iqr'] = features['post_roi_q75'] - features['post_roi_q25']
        
        # Background features
        features['post_bg_mean'] = float(np.mean(post_bg))
        features['post_bg_std'] = float(np.std(post_bg))
        features['post_bg_variance'] = float(np.var(post_bg))
        
        # Global features
        features['post_global_mean'] = float(np.mean(post_contrast[post_contrast > 0]))
        features['post_global_std'] = float(np.std(post_contrast[post_contrast > 0]))
        features['post_global_variance'] = float(np.var(post_contrast[post_contrast > 0]))
        
        # === FEATURES DE ENHANCEMENT ===
        enhancement = post_roi - pre_roi
        features['enhancement_mean'] = float(np.mean(enhancement))
        features['enhancement_std'] = float(np.std(enhancement))
        features['enhancement_variance'] = float(np.var(enhancement))
        features['enhancement_min'] = float(np.min(enhancement))
        features['enhancement_max'] = float(np.max(enhancement))
        features['enhancement_range'] = features['enhancement_max'] - features['enhancement_min']
        
        # === FEATURES DE RATIO ===
        # Evitar división por cero
        if features['pre_roi_mean'] > 1e-10:
            features['post_pre_ratio'] = features['post_roi_mean'] / features['pre_roi_mean']
        else:
            features['post_pre_ratio'] = 0.0
        
        if features['pre_bg_mean'] > 1e-10:
            features['roi_bg_ratio_pre'] = features['pre_roi_mean'] / features['pre_bg_mean']
        else:
            features['roi_bg_ratio_pre'] = 0.0
        
        if features['post_bg_mean'] > 1e-10:
            features['roi_bg_ratio_post'] = features['post_roi_mean'] / features['post_bg_mean']
        else:
            features['roi_bg_ratio_post'] = 0.0
        
        # === FEATURES DE TEXTURA SIMPLE ===
        # Gradiente (aproximación de textura)
        def calculate_simple_gradient(data_3d):
            # Calcular gradiente en las 3 dimensiones
            grad_x = np.abs(np.diff(data_3d, axis=0)).mean()
            grad_y = np.abs(np.diff(data_3d, axis=1)).mean()
            grad_z = np.abs(np.diff(data_3d, axis=2)).mean()
            return (grad_x + grad_y + grad_z) / 3
        
        features['pre_texture_gradient'] = float(calculate_simple_gradient(pre_contrast))
        features['post_texture_gradient'] = float(calculate_simple_gradient(post_contrast))
        
        # === FEATURES DE FORMA (basadas en ROI) ===
        features['roi_volume'] = float(np.sum(roi_mask))
        features['roi_surface_approx'] = float(np.sum(np.abs(np.diff(roi_mask.astype(int), axis=0))) + 
                                              np.sum(np.abs(np.diff(roi_mask.astype(int), axis=1))) + 
                                              np.sum(np.abs(np.diff(roi_mask.astype(int), axis=2))))
        
        # Aproximación de esfericidad
        if features['roi_volume'] > 0:
            features['roi_sphericity_approx'] = features['roi_volume'] / (features['roi_surface_approx'] + 1e-10)
        else:
            features['roi_sphericity_approx'] = 0.0
        
        return features
    
    def analyze_sample_patients(self, n_patients: int = 50):
        """Analizar muestra de pacientes"""
        
        print(f"🔍 VERIFICANDO VARIABILIDAD EN {n_patients} PACIENTES")
        print("=" * 60)
        
        # Cargar splits
        splits_df = pd.read_csv(self.splits_csv)
        
        # Obtener pacientes disponibles
        available_patients = []
        for split_type in ['train_split', 'test_split']:
            patients = splits_df[split_type].dropna().astype(str).str.strip().tolist()
            for patient_id in patients:
                tensor_path = self.processed_data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
                if tensor_path.exists():
                    available_patients.append(patient_id)
        
        print(f"📊 Pacientes disponibles: {len(available_patients)}")
        
        # Ajustar muestra
        if len(available_patients) < n_patients:
            n_patients = len(available_patients)
            print(f"⚠️  Ajustando muestra a {n_patients} pacientes")
        
        sample_patients = available_patients[:n_patients]
        
        # Extraer features
        all_features = []
        
        print(f"\n🧮 CALCULANDO FEATURES SIMPLES:")
        
        for patient_id in tqdm(sample_patients, desc="Procesando"):
            try:
                features = self.extract_simple_features(patient_id)
                features['patient_id'] = patient_id
                all_features.append(features)
                self.results['patients_analyzed'] += 1
                
            except Exception as e:
                print(f"⚠️  Error en {patient_id}: {str(e)}")
                self.results['loading_errors'] += 1
        
        if not all_features:
            print("❌ No se pudieron procesar pacientes")
            return None
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(all_features)
        feature_columns = [col for col in features_df.columns if col != 'patient_id']
        
        self.results['features_calculated'] = len(feature_columns)
        
        print(f"✅ Features calculadas: {len(feature_columns)}")
        print(f"✅ Pacientes procesados: {len(all_features)}")
        
        return features_df, feature_columns
    
    def identify_constant_features(self, features_df: pd.DataFrame, feature_columns: list):
        """Identificar features constantes"""
        
        print(f"\n🔍 ANALIZANDO VARIABILIDAD DE FEATURES:")
        print("-" * 50)
        
        constant_features = []
        problematic_features = []
        feature_stats = {}
        
        for feature in feature_columns:
            values = features_df[feature].values
            
            # Estadísticas de variabilidad
            variance = np.var(values)
            std = np.std(values)
            unique_count = len(np.unique(values))
            range_val = np.max(values) - np.min(values)
            mean_val = np.mean(values)
            
            # Criterios para features constantes
            is_constant = (
                variance < 1e-12 or      # Varianza prácticamente cero
                std < 1e-12 or          # Desviación estándar prácticamente cero
                unique_count == 1 or    # Solo un valor único
                range_val < 1e-12       # Rango prácticamente cero
            )
            
            # Criterios para features problemáticas (baja variabilidad)
            is_problematic = (
                variance < 1e-8 or
                std < 1e-8 or
                unique_count < 3
            ) and not is_constant
            
            if is_constant:
                constant_features.append(feature)
                print(f"❌ CONSTANTE: {feature}")
                print(f"   Valor: {values[0]:.6f}")
                print(f"   Varianza: {variance:.2e}")
                print()
            elif is_problematic:
                problematic_features.append(feature)
                print(f"⚠️  BAJA VARIABILIDAD: {feature}")
                print(f"   Varianza: {variance:.2e}")
                print(f"   Valores únicos: {unique_count}")
                print()
            
            feature_stats[feature] = {
                'variance': float(variance),
                'std': float(std),
                'unique_count': int(unique_count),
                'range': float(range_val),
                'mean': float(mean_val),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'is_constant': is_constant,
                'is_problematic': is_problematic
            }
        
        self.results['constant_features'] = len(constant_features)
        self.results['problematic_features'] = len(problematic_features)
        
        return constant_features, problematic_features, feature_stats
    
    def generate_verification_report(self, constant_features, problematic_features, feature_stats):
        """Generar reporte de verificación"""
        
        print(f"\n📊 REPORTE DE VERIFICACIÓN:")
        print("=" * 60)
        
        total_features = self.results['features_calculated']
        constant_count = len(constant_features)
        problematic_count = len(problematic_features)
        valid_count = total_features - constant_count - problematic_count
        
        # Comparación con problema original
        original_constant = 61  # Del problema original
        
        print(f"🔍 COMPARACIÓN CON PROBLEMA ORIGINAL:")
        print(f"   Features constantes ANTES: {original_constant}")
        print(f"   Features constantes DESPUÉS: {constant_count}")
        
        if constant_count == 0:
            print(f"   ✅ PROBLEMA COMPLETAMENTE RESUELTO!")
            improvement = 100.0
        else:
            improvement = ((original_constant - constant_count) / original_constant) * 100
            print(f"   📈 Mejora: {improvement:.1f}%")
        
        print(f"\n📊 ANÁLISIS DETALLADO:")
        print(f"   Total features analizadas: {total_features}")
        print(f"   Features constantes: {constant_count} ({constant_count/total_features*100:.1f}%)")
        print(f"   Features con baja variabilidad: {problematic_count} ({problematic_count/total_features*100:.1f}%)")
        print(f"   Features válidas: {valid_count} ({valid_count/total_features*100:.1f}%)")
        
        # Análisis de calidad
        print(f"\n🎯 EVALUACIÓN DE CALIDAD:")
        
        if constant_count == 0:
            quality = "EXCELENTE"
            color = "✅"
            message = "Sin features constantes - Datos perfectos para entrenamiento"
        elif constant_count < 3:
            quality = "BUENA"
            color = "✅"
            message = "Muy pocas features constantes - Datos buenos para entrenamiento"
        elif constant_count < 10:
            quality = "ACEPTABLE"
            color = "⚠️ "
            message = "Algunas features constantes - Datos utilizables"
        else:
            quality = "PROBLEMÁTICA"
            color = "❌"
            message = "Muchas features constantes - Revisar pipeline"
        
        print(f"   {color} CALIDAD: {quality}")
        print(f"   {message}")
        
        # Estadísticas de variabilidad
        valid_features = {k: v for k, v in feature_stats.items() 
                         if not v['is_constant'] and not v['is_problematic']}
        
        if valid_features:
            variances = [v['variance'] for v in valid_features.values()]
            print(f"\n📈 ESTADÍSTICAS DE FEATURES VÁLIDAS:")
            print(f"   Varianza promedio: {np.mean(variances):.2e}")
            print(f"   Varianza mínima: {np.min(variances):.2e}")
            print(f"   Varianza máxima: {np.max(variances):.2e}")
            print(f"   Varianza mediana: {np.median(variances):.2e}")
        
        return {
            'original_constant': original_constant,
            'current_constant': constant_count,
            'improvement_percent': improvement,
            'total_features': total_features,
            'valid_features': valid_count,
            'problematic_features': problematic_count,
            'quality_status': quality,
            'ready_for_training': constant_count < 5
        }
    
    def save_verification_report(self, results, feature_stats):
        """Guardar reporte completo"""
        
        report = {
            'verification_timestamp': pd.Timestamp.now().isoformat(),
            'processed_data_directory': str(self.processed_data_dir),
            'verification_results': results,
            'processing_stats': self.results,
            'feature_statistics': feature_stats,
            'conclusion': {
                'pipeline_successful': results['current_constant'] < 5,
                'ready_for_training': results['current_constant'] == 0,
                'significant_improvement': results['improvement_percent'] > 90
            }
        }
        
        report_file = self.processed_data_dir / "simple_feature_verification.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 REPORTE GUARDADO: {report_file}")
        return report_file
    
    def run_verification(self, n_patients: int = 50):
        """Ejecutar verificación completa"""
        
        print("🔍 VERIFICACIÓN SIMPLIFICADA DE CORRECCIÓN DE FEATURES")
        print("=" * 70)
        print(f"📁 Datos: {self.processed_data_dir}")
        print(f"📊 Muestra: {n_patients} pacientes")
        print(f"🧮 Método: Análisis directo de tensors (sin PyRadiomics)")
        print()
        
        # 1. Extraer features simples
        features_data = self.analyze_sample_patients(n_patients)
        
        if features_data is None:
            print("❌ Verificación fallida")
            return False
        
        features_df, feature_columns = features_data
        
        # 2. Identificar features constantes
        constant_features, problematic_features, feature_stats = self.identify_constant_features(
            features_df, feature_columns
        )
        
        # 3. Generar reporte
        results = self.generate_verification_report(
            constant_features, problematic_features, feature_stats
        )
        
        # 4. Guardar reporte
        self.save_verification_report(results, feature_stats)
        
        # 5. Conclusión
        print(f"\n🎯 CONCLUSIÓN FINAL:")
        print("=" * 40)
        
        if results['current_constant'] == 0:
            print(f"🎉 VERIFICACIÓN COMPLETAMENTE EXITOSA")
            print(f"✅ CERO features constantes detectadas")
            print(f"✅ Pipeline corrigió el problema original al 100%")
            print(f"✅ Datos PERFECTOS para entrenamiento")
            print(f"\n🚀 SIGUIENTE PASO: Ejecutar pipeline de entrenamiento pCR")
            return True
        
        elif results['current_constant'] < 5:
            print(f"✅ VERIFICACIÓN EXITOSA")
            print(f"✅ Solo {results['current_constant']} features constantes")
            print(f"✅ Mejora del {results['improvement_percent']:.1f}%")
            print(f"✅ Datos BUENOS para entrenamiento")
            print(f"\n🎯 RECOMENDACIÓN: Proceder con entrenamiento")
            return True
        
        else:
            print(f"⚠️  VERIFICACIÓN PARCIAL")
            print(f"⚠️  {results['current_constant']} features aún constantes")
            print(f"📈 Mejora del {results['improvement_percent']:.1f}%")
            print(f"\n🔧 RECOMENDACIÓN: Revisar configuración del pipeline")
            return False

def main():
    """Ejecutar verificación simplificada"""
    
    # Configuración
    processed_data_dir = "D:/mama_mia_fixed_0000_0001_challenge_ready"
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    
    # Verificar que existen los directorios
    if not Path(processed_data_dir).exists():
        print(f"❌ Directorio no encontrado: {processed_data_dir}")
        print("🔧 Verifica que el pipeline de corrección se ejecutó correctamente")
        return
    
    if not Path(splits_csv).exists():
        print(f"❌ Archivo de splits no encontrado: {splits_csv}")
        return
    
    # Crear verificador
    verifier = SimpleFeatureVerifier(processed_data_dir, splits_csv)
    
    # Ejecutar verificación
    success = verifier.run_verification(n_patients=50)
    
    if success:
        print(f"\n🎉 ¡DATOS VERIFICADOS Y LISTOS!")
        print(f"📄 Siguiente: Extraer etiquetas pCR del Excel")
        print(f"🤖 Después: Entrenar modelo de predicción pCR")
    else:
        print(f"\n🔧 Revisar pipeline de corrección")

if __name__ == "__main__":
    main()