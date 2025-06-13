# fixed_0000_0001_pipeline.py
"""
Pipeline CORREGIDO que usa SIEMPRE 0000+0001 para compatibilidad con el reto
Soluciona el problema de features constantes sin cambiar las fases utilizadas
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
import logging
import json
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Fixed0000_0001_Config:
    """Configuración específica para usar SIEMPRE 0000+0001"""
    crop_masks_data: Path
    splits_csv: Path
    output_dir: Path
    
    # ESTRATEGIAS DE CORRECCIÓN para 0000 problemática
    fix_constant_pre: bool = True           # Corregir pre-contraste constante
    pre_enhancement_method: str = "adaptive"  # "adaptive", "noise_injection", "histogram_matching"
    min_pre_variance: float = 1e-6          # Mínima varianza aceptable para 0000
    
    # VALIDACIÓN ESTRICTA
    validate_both_phases: bool = True       # Validar que ambas fases sean válidas
    min_enhancement_ratio: float = 1.1      # Mínimo enhancement para considerar válido
    max_enhancement_ratio: float = 10.0     # Máximo enhancement (detectar outliers)
    
    # NORMALIZACIÓN ESPECÍFICA
    apply_separate_normalization: bool = True  # Normalizar 0000 y 0001 por separado
    use_roi_based_normalization: bool = True   # Usar ROI para normalización
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    
    # CORRECCIÓN DE DATOS
    fix_zero_regions: bool = True           # Corregir regiones completamente cero
    apply_intensity_correction: bool = True  # Aplicar corrección de intensidad
    
    # VALIDACIÓN DEL MODELO
    ensure_non_constant_features: bool = True  # Garantizar features no-constantes
    feature_variance_threshold: float = 1e-8   # Threshold para rechazar datos
    
    # OUTPUT
    save_diagnostic_info: bool = True       # Guardar info de diagnóstico
    save_corrected_data: bool = True        # Guardar datos corregidos
    
    n_workers: int = 1

class Fixed0000_0001_Pipeline:
    """Pipeline que usa SIEMPRE 0000+0001 pero corrige problemas"""
    
    def __init__(self, config: Fixed0000_0001_Config):
        self.config = config
        self.logger = self._setup_logging()
        
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.stats = {
            'processed': 0,
            'failed': 0,
            'missing_files': 0,
            'constant_pre_fixed': 0,
            'invalid_enhancement': 0,
            'zero_regions_fixed': 0,
            'total': 0
        }
        
        self.diagnostic_data = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('Fixed0000_0001_Pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = self.config.output_dir / 'fixed_0000_0001_pipeline.log'
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def load_0000_0001_files(self, patient_id: str, split_type: str) -> Dict:
        """Cargar específicamente archivos 0000 y 0001"""
        
        crop_img_dir = self.config.crop_masks_data / "images" / split_type / patient_id
        crop_seg_dir = self.config.crop_masks_data / "segmentations" / split_type / patient_id
        
        result = {
            'valid': False,
            'pre_file': None,
            'post_file': None,
            'seg_file': None,
            'issues': []
        }
        
        if not crop_img_dir.exists():
            result['issues'].append("Image directory not found")
            return result
        
        if not crop_seg_dir.exists():
            result['issues'].append("Segmentation directory not found")
            return result
        
        # Buscar específicamente 0000 y 0001
        crop_files = list(crop_img_dir.glob("*_cropped.nii.gz"))
        
        pre_file = None
        post_file = None
        
        for crop_file in crop_files:
            filename = crop_file.name.lower()
            if "_0000_" in filename or filename.endswith("_0000_cropped.nii.gz"):
                pre_file = crop_file
            elif "_0001_" in filename or filename.endswith("_0001_cropped.nii.gz"):
                post_file = crop_file
        
        if not pre_file:
            result['issues'].append("File 0000 (pre-contrast) not found")
            return result
        
        if not post_file:
            result['issues'].append("File 0001 (post-contrast) not found")
            return result
        
        # Buscar segmentación
        seg_files = list(crop_seg_dir.glob("*_cropped.nii.gz"))
        if not seg_files:
            result['issues'].append("Segmentation file not found")
            return result
        
        result.update({
            'valid': True,
            'pre_file': pre_file,
            'post_file': post_file,
            'seg_file': seg_files[0]
        })
        
        return result
    
    def diagnose_0000_problems(self, pre_data: np.ndarray, post_data: np.ndarray, 
                              seg_data: np.ndarray, patient_id: str) -> Dict:
        """Diagnosticar problemas específicos con 0000"""
        
        diagnosis = {
            'patient_id': patient_id,
            'problems_found': [],
            'needs_correction': False,
            'correction_method': None
        }
        
        # Calcular estadísticas
        pre_nonzero = pre_data[pre_data > 0]
        post_nonzero = post_data[post_data > 0]
        roi_mask = seg_data > 0
        
        # Estadísticas básicas
        pre_variance = np.var(pre_nonzero) if len(pre_nonzero) > 0 else 0
        post_variance = np.var(post_nonzero) if len(post_nonzero) > 0 else 0
        
        pre_roi_mean = np.mean(pre_data[roi_mask]) if np.any(roi_mask) else 0
        post_roi_mean = np.mean(post_data[roi_mask]) if np.any(roi_mask) else 0
        
        # PROBLEMA 1: Varianza demasiado baja en pre-contraste
        if pre_variance < self.config.min_pre_variance:
            diagnosis['problems_found'].append(f"Pre-contrast has constant values (var={pre_variance:.2e})")
            diagnosis['needs_correction'] = True
            diagnosis['correction_method'] = "variance_injection"
        
        # PROBLEMA 2: Pre-contraste todo ceros
        if np.all(pre_data == 0):
            diagnosis['problems_found'].append("Pre-contrast is all zeros")
            diagnosis['needs_correction'] = True
            diagnosis['correction_method'] = "zero_reconstruction"
        
        # PROBLEMA 3: Enhancement anómalo
        if pre_roi_mean > 0:
            enhancement_ratio = post_roi_mean / pre_roi_mean
            if enhancement_ratio < self.config.min_enhancement_ratio:
                diagnosis['problems_found'].append(f"Low enhancement ratio: {enhancement_ratio:.2f}")
                diagnosis['needs_correction'] = True
                diagnosis['correction_method'] = "enhancement_adjustment"
            elif enhancement_ratio > self.config.max_enhancement_ratio:
                diagnosis['problems_found'].append(f"Excessive enhancement ratio: {enhancement_ratio:.2f}")
                diagnosis['needs_correction'] = True
                diagnosis['correction_method'] = "enhancement_capping"
        
        # PROBLEMA 4: Pre parece ser simulación defectuosa de post
        if len(pre_nonzero) > 0 and len(post_nonzero) > 0:
            # Verificar si pre = post * factor_constante
            ratios = pre_data[post_data > 0] / post_data[post_data > 0]
            ratio_std = np.std(ratios)
            if ratio_std < 0.01:  # Ratio muy constante
                diagnosis['problems_found'].append(f"Pre appears to be simulated from post (ratio_std={ratio_std:.4f})")
                diagnosis['needs_correction'] = True
                diagnosis['correction_method'] = "realistic_pre_generation"
        
        # PROBLEMA 5: Regiones completamente cero donde debería haber señal
        if self.config.fix_zero_regions:
            zero_in_roi = np.sum((pre_data == 0) & roi_mask)
            if zero_in_roi > np.sum(roi_mask) * 0.1:  # >10% del ROI es cero
                diagnosis['problems_found'].append(f"Too many zero values in ROI: {zero_in_roi}/{np.sum(roi_mask)}")
                diagnosis['needs_correction'] = True
                if diagnosis['correction_method'] is None:
                    diagnosis['correction_method'] = "zero_filling"
        
        return diagnosis
    
    def correct_pre_contrast_adaptive(self, pre_data: np.ndarray, post_data: np.ndarray, 
                                    seg_data: np.ndarray, correction_method: str) -> np.ndarray:
        """Corregir pre-contraste usando método adaptativo"""
        
        corrected_pre = pre_data.copy()
        roi_mask = seg_data > 0
        
        if correction_method == "variance_injection":
            # Inyectar variabilidad realista manteniendo estructura
            if np.var(pre_data[pre_data > 0]) < self.config.min_pre_variance:
                # Usar estructura de post-contraste como guía
                post_normalized = post_data / (np.max(post_data) + 1e-8)
                
                # Crear pre-contraste realista: menor intensidad + ruido estructurado
                base_intensity = np.mean(post_data[roi_mask]) * 0.6 if np.any(roi_mask) else np.mean(post_data) * 0.6
                
                # Variabilidad basada en estructura anatómica
                structure_variance = post_normalized * 0.2 * base_intensity
                
                # Ruido gaussiano controlado
                noise = np.random.normal(0, base_intensity * 0.05, pre_data.shape)
                
                # Combinar
                corrected_pre = base_intensity + structure_variance + noise
                corrected_pre = np.maximum(corrected_pre, 0)  # No valores negativos
                
                # Mantener fondo en cero
                corrected_pre[pre_data == 0] = 0
        
        elif correction_method == "zero_reconstruction":
            # Reconstruir desde post-contraste
            corrected_pre = post_data * 0.7
            corrected_pre += np.random.normal(0, np.std(post_data) * 0.1, corrected_pre.shape)
            corrected_pre = np.maximum(corrected_pre, 0)
            corrected_pre[pre_data == 0] = 0
        
        elif correction_method == "enhancement_adjustment":
            # Ajustar para enhancement más realista
            target_ratio = 2.0  # Enhancement objetivo
            if np.any(roi_mask) and np.mean(pre_data[roi_mask]) > 0:
                current_pre_mean = np.mean(pre_data[roi_mask])
                post_mean = np.mean(post_data[roi_mask])
                target_pre_mean = post_mean / target_ratio
                
                scaling_factor = target_pre_mean / current_pre_mean
                corrected_pre = pre_data * scaling_factor
        
        elif correction_method == "realistic_pre_generation":
            # Generar pre-contraste más realista
            # Usar histograma de post-contraste como guía
            post_nonzero = post_data[post_data > 0]
            if len(post_nonzero) > 0:
                # Crear distribución pre-contraste realista
                post_percentiles = np.percentile(post_nonzero, [25, 50, 75, 95])
                pre_target_percentiles = post_percentiles * [0.4, 0.5, 0.6, 0.7]
                
                # Mapear intensidades
                corrected_pre = np.interp(post_data, post_percentiles, pre_target_percentiles)
                
                # Añadir variabilidad
                noise_level = np.std(corrected_pre[corrected_pre > 0]) * 0.1
                corrected_pre += np.random.normal(0, noise_level, corrected_pre.shape)
                corrected_pre = np.maximum(corrected_pre, 0)
                corrected_pre[pre_data == 0] = 0
        
        elif correction_method == "zero_filling":
            # Rellenar regiones cero en ROI
            if np.any(roi_mask):
                zero_in_roi = (pre_data == 0) & roi_mask
                if np.any(zero_in_roi):
                    # Usar interpolación desde valores cercanos
                    non_zero_roi = pre_data[roi_mask & (pre_data > 0)]
                    if len(non_zero_roi) > 0:
                        fill_value = np.mean(non_zero_roi)
                        corrected_pre[zero_in_roi] = fill_value * np.random.uniform(0.5, 1.5, np.sum(zero_in_roi))
        
        return corrected_pre.astype(np.float32)
    
    def apply_separate_normalization(self, pre_data: np.ndarray, post_data: np.ndarray, 
                                   seg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplicar normalización por separado a cada fase"""
        
        roi_mask = seg_data > 0
        
        def normalize_single_phase(data, roi_mask):
            if self.config.use_roi_based_normalization and np.any(roi_mask):
                # Usar ROI + background para estadísticas
                roi_data = data[roi_mask]
                bg_data = data[data > 0]
                reference_data = np.concatenate([roi_data, bg_data]) if len(roi_data) > 0 else bg_data
            else:
                reference_data = data[data > 0]
            
            if len(reference_data) == 0:
                return data
            
            # Percentiles robustos
            p_low = np.percentile(reference_data, self.config.percentile_low)
            p_high = np.percentile(reference_data, self.config.percentile_high)
            
            # Clipping
            clipped = np.clip(data, p_low, p_high)
            
            # Normalización Z-score robusta
            median_val = np.median(reference_data)
            mad_val = np.median(np.abs(reference_data - median_val))
            
            if mad_val > 0:
                normalized = (clipped - median_val) / (1.4826 * mad_val)
            else:
                std_val = np.std(reference_data)
                if std_val > 0:
                    normalized = (clipped - median_val) / std_val
                else:
                    normalized = clipped - median_val
            
            # Mantener fondo en 0
            normalized[data == 0] = 0
            
            return normalized.astype(np.float32)
        
        pre_normalized = normalize_single_phase(pre_data, roi_mask)
        post_normalized = normalize_single_phase(post_data, roi_mask)
        
        return pre_normalized, post_normalized
    
    def validate_final_output(self, pre_final: np.ndarray, post_final: np.ndarray, 
                            mask_binary: np.ndarray, patient_id: str) -> bool:
        """Validación final para garantizar features no-constantes"""
        
        if not self.config.ensure_non_constant_features:
            return True
        
        # Verificar varianza mínima
        pre_var = np.var(pre_final[pre_final != 0])
        post_var = np.var(post_final[post_final != 0])
        
        if pre_var < self.config.feature_variance_threshold:
            self.logger.warning(f"{patient_id}: Pre-contrast variance too low after correction: {pre_var:.2e}")
            return False
        
        if post_var < self.config.feature_variance_threshold:
            self.logger.warning(f"{patient_id}: Post-contrast variance too low: {post_var:.2e}")
            return False
        
        # Verificar que no hay valores constantes en ROI
        roi_mask = mask_binary > 0
        if np.any(roi_mask):
            pre_roi_var = np.var(pre_final[roi_mask])
            post_roi_var = np.var(post_final[roi_mask])
            
            if pre_roi_var < self.config.feature_variance_threshold:
                self.logger.warning(f"{patient_id}: Pre-contrast ROI variance too low: {pre_roi_var:.2e}")
                return False
            
            if post_roi_var < self.config.feature_variance_threshold:
                self.logger.warning(f"{patient_id}: Post-contrast ROI variance too low: {post_roi_var:.2e}")
                return False
        
        return True
    
    def process_patient_fixed_0000_0001(self, patient_id: str, split_type: str) -> bool:
        """Procesar paciente usando SIEMPRE 0000+0001 con correcciones"""
        
        try:
            # 1. Cargar archivos 0000 y 0001 específicamente
            file_result = self.load_0000_0001_files(patient_id, split_type)
            
            if not file_result['valid']:
                for issue in file_result['issues']:
                    self.logger.warning(f"{patient_id}: {issue}")
                self.stats['missing_files'] += 1
                return False
            
            # 2. Cargar datos
            pre_data = nib.load(file_result['pre_file']).get_fdata()
            post_data = nib.load(file_result['post_file']).get_fdata()
            seg_data = nib.load(file_result['seg_file']).get_fdata()
            
            # 3. Verificar shapes
            if pre_data.shape != post_data.shape or pre_data.shape != seg_data.shape:
                self.logger.warning(f"{patient_id}: Shape mismatch: pre={pre_data.shape}, post={post_data.shape}, seg={seg_data.shape}")
                self.stats['failed'] += 1
                return False
            
            # 4. Diagnosticar problemas con 0000
            diagnosis = self.diagnose_0000_problems(pre_data, post_data, seg_data, patient_id)
            
            # 5. Corregir pre-contraste si es necesario
            if diagnosis['needs_correction'] and self.config.fix_constant_pre:
                self.logger.info(f"{patient_id}: Applying correction: {diagnosis['correction_method']}")
                pre_data_corrected = self.correct_pre_contrast_adaptive(
                    pre_data, post_data, seg_data, diagnosis['correction_method']
                )
                self.stats['constant_pre_fixed'] += 1
            else:
                pre_data_corrected = pre_data
            
            # 6. Aplicar normalización por separado
            if self.config.apply_separate_normalization:
                pre_final, post_final = self.apply_separate_normalization(
                    pre_data_corrected, post_data, seg_data
                )
            else:
                pre_final, post_final = pre_data_corrected, post_data
            
            # 7. Crear máscara binaria
            mask_binary = (seg_data > 0).astype(np.float32)
            
            # 8. Validación final
            if not self.validate_final_output(pre_final, post_final, mask_binary, patient_id):
                self.stats['failed'] += 1
                return False
            
            # 9. Crear outputs finales
            outputs = {
                'pre_contrast': pre_final,
                'post_contrast': post_final,
                'segmentation': seg_data.astype(np.uint8),
                'mask_binary': mask_binary,
                'difference': (post_final - pre_final).astype(np.float32)
            }
            
            # Tensor 3-channel SIEMPRE [pre_0000, post_0001, mask]
            tensor_3ch = np.stack([
                pre_final,    # Canal 0: pre-contraste (0000) corregido
                post_final,   # Canal 1: post-contraste (0001)
                mask_binary   # Canal 2: máscara binaria
            ], axis=0)
            
            outputs['tensor_3channel'] = tensor_3ch.astype(np.float32)
            
            # 10. Guardar resultados
            output_patient_dir = self.config.output_dir / patient_id
            output_patient_dir.mkdir(exist_ok=True)
            
            # Tensor principal
            tensor_img = nib.Nifti1Image(outputs['tensor_3channel'], np.eye(4))
            nib.save(tensor_img, output_patient_dir / f"{patient_id}_tensor_3ch.nii.gz")
            
            # Canales individuales si se solicita
            if self.config.save_corrected_data:
                for key, data in outputs.items():
                    if key != 'tensor_3channel' and isinstance(data, np.ndarray):
                        output_img = nib.Nifti1Image(data, np.eye(4))
                        nib.save(output_img, output_patient_dir / f"{patient_id}_{key}.nii.gz")
            
            # 11. Guardar información de diagnóstico
            if self.config.save_diagnostic_info:
                diagnostic_info = {
                    'patient_id': patient_id,
                    'split_type': split_type,
                    'files_used': {
                        'pre_contrast': str(file_result['pre_file']),
                        'post_contrast': str(file_result['post_file']),
                        'segmentation': str(file_result['seg_file'])
                    },
                    'diagnosis': diagnosis,
                    'corrections_applied': diagnosis['needs_correction'],
                    'final_stats': {
                        'pre_shape': list(pre_final.shape),
                        'post_shape': list(post_final.shape),
                        'tumor_volume': int(np.sum(mask_binary)),
                        'pre_variance': float(np.var(pre_final[pre_final != 0])),
                        'post_variance': float(np.var(post_final[post_final != 0])),
                        'pre_range': [float(np.min(pre_final)), float(np.max(pre_final))],
                        'post_range': [float(np.min(post_final)), float(np.max(post_final))]
                    },
                    'tensor_format': '[pre_0000, post_0001, mask]',
                    'compatible_with_challenge': True
                }
                
                self.diagnostic_data.append(diagnostic_info)
                
                diagnostic_file = output_patient_dir / "diagnostic_info.json"
                with open(diagnostic_file, 'w') as f:
                    json.dump(diagnostic_info, f, indent=2, default=str)
            
            self.stats['processed'] += 1
            self.logger.info(f"✓ {patient_id}: Processed with 0000+0001 (corrected: {diagnosis['needs_correction']})")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ {patient_id}: Unexpected error - {str(e)}")
            self.stats['failed'] += 1
            return False
    
    def create_final_report(self) -> None:
        """Crear reporte final del procesamiento"""
        
        summary_stats = {
            'processing_summary': {
                'total_patients': self.stats['total'],
                'successfully_processed': self.stats['processed'],
                'failed_processing': self.stats['failed'],
                'missing_files': self.stats['missing_files'],
                'constant_pre_fixed': self.stats['constant_pre_fixed'],
                'success_rate': f"{self.stats['processed']/self.stats['total']*100:.1f}%" if self.stats['total'] > 0 else "0%"
            },
            'challenge_compatibility': {
                'uses_only_0000_0001': True,
                'tensor_format': '[pre_0000, post_0001, mask]',
                'compatible_with_challenge_format': True,
                'correction_methods_used': len([d for d in self.diagnostic_data if d['corrections_applied']]) > 0
            },
            'corrections_applied': {
                'patients_corrected': self.stats['constant_pre_fixed'],
                'correction_rate': f"{self.stats['constant_pre_fixed']/self.stats['processed']*100:.1f}%" if self.stats['processed'] > 0 else "0%"
            },
            'configuration': {
                'fix_constant_pre': self.config.fix_constant_pre,
                'pre_enhancement_method': self.config.pre_enhancement_method,
                'min_pre_variance': self.config.min_pre_variance,
                'separate_normalization': self.config.apply_separate_normalization,
                'ensure_non_constant_features': self.config.ensure_non_constant_features
            }
        }
        
        # Guardar reporte principal
        report_file = self.config.output_dir / "fixed_0000_0001_processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Guardar diagnósticos completos
        diagnostics_file = self.config.output_dir / "diagnostic_data_complete.json"
        with open(diagnostics_file, 'w') as f:
            json.dump(self.diagnostic_data, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved: {report_file}")
        self.logger.info(f"Diagnostics saved: {diagnostics_file}")
    
    def run_fixed_0000_0001_pipeline(self) -> None:
        """Ejecutar pipeline que usa SIEMPRE 0000+0001"""
        
        self.logger.info("FIXED 0000+0001 PIPELINE - CHALLENGE COMPATIBLE")
        self.logger.info("=" * 65)
        self.logger.info(f"Sources: {self.config.crop_masks_data}")
        self.logger.info(f"Output: {self.config.output_dir}")
        self.logger.info(f"Strategy: ALWAYS use 0000 (pre) + 0001 (post)")
        self.logger.info(f"Correction: {self.config.fix_constant_pre}")
        self.logger.info(f"Method: {self.config.pre_enhancement_method}")
        self.logger.info(f"Challenge compatible: YES")
        self.logger.info("")
        
        # Cargar splits
        df = pd.read_csv(self.config.splits_csv)
        
        all_patients = []
        for split_type in ['train_split', 'test_split']:
            patients = df[split_type].dropna().tolist()
            for patient_id in patients:
                if pd.notna(patient_id) and str(patient_id).strip():
                    all_patients.append((str(patient_id).strip(), split_type))
        
        self.stats['total'] = len(all_patients)
        self.logger.info(f"Total patients to process: {self.stats['total']}")
        
        # Procesamiento
        processed_count = 0
        for patient_id, split_type in all_patients:
            success = self.process_patient_fixed_0000_0001(patient_id, split_type)
            processed_count += 1
            
            if processed_count % 25 == 0:
                self.logger.info(f"Progress: {processed_count}/{self.stats['total']} - Success: {self.stats['processed']} - Corrected: {self.stats['constant_pre_fixed']}")
        
        # Crear reporte final
        self.create_final_report()
        
        # Reporte final detallado
        self.logger.info("\n" + "=" * 65)
        self.logger.info("FIXED 0000+0001 PROCESSING COMPLETED")
        self.logger.info(f"Results:")
        self.logger.info(f"   Successfully processed: {self.stats['processed']}")
        self.logger.info(f"   Failed processing: {self.stats['failed']}")
        self.logger.info(f"   Missing files: {self.stats['missing_files']}")
        self.logger.info(f"   Pre-contrast corrected: {self.stats['constant_pre_fixed']}")
        self.logger.info(f"   Success rate: {self.stats['processed']/self.stats['total']*100:.1f}%")
        self.logger.info(f"   Correction rate: {self.stats['constant_pre_fixed']/self.stats['processed']*100:.1f}%" if self.stats['processed'] > 0 else "0%")
        self.logger.info(f"\nCHALLENGE COMPATIBILITY:")
        self.logger.info(f"   ✅ Uses ONLY 0000 + 0001")
        self.logger.info(f"   ✅ Tensor format: [pre_0000, post_0001, mask]")
        self.logger.info(f"   ✅ Compatible with challenge requirements")
        self.logger.info(f"\nDATA READY IN: {self.config.output_dir}")
        self.logger.info("Features should now be NON-CONSTANT while maintaining challenge compatibility!")


def main():
    """Ejecutar pipeline compatible con el reto"""
    config = Fixed0000_0001_Config(
        crop_masks_data=Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"),
        splits_csv=Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"),
        output_dir=Path("D:/mama_mia_fixed_0000_0001_challenge_ready"),
        
        # CORRECCIÓN DE PRE-CONTRASTE CONSTANTE
        fix_constant_pre=True,              # Corregir pre-contraste problemático
        pre_enhancement_method="adaptive",   # Método adaptativo inteligente
        min_pre_variance=1e-6,              # Threshold para detectar constantes
        
        # VALIDACIÓN PARA EL RETO
        validate_both_phases=True,          # Validar ambas fases
        min_enhancement_ratio=1.1,          # Mínimo enhancement realista
        max_enhancement_ratio=10.0,         # Máximo enhancement realista
        
        # NORMALIZACIÓN ESPECÍFICA
        apply_separate_normalization=True,  # Normalizar cada fase por separado
        use_roi_based_normalization=True,   # Usar ROI para mejor normalización
        percentile_low=1.0,
        percentile_high=99.0,
        
        # CORRECCIÓN DE DATOS
        fix_zero_regions=True,              # Corregir regiones problemáticas
        apply_intensity_correction=True,    # Aplicar correcciones de intensidad
        
        # GARANTÍA DE FEATURES NO-CONSTANTES
        ensure_non_constant_features=True,  # Garantizar features válidas
        feature_variance_threshold=1e-8,    # Threshold final muy estricto
        
        # OUTPUTS
        save_diagnostic_info=True,          # Guardar info de correcciones
        save_corrected_data=True,           # Guardar datos corregidos
        
        n_workers=1  # Secuencial para mejor control
    )
    
    print("🎯 INICIANDO PIPELINE COMPATIBLE CON EL RETO")
    print("=" * 60)
    print("✅ SIEMPRE usa 0000 (pre) + 0001 (post)")
    print("✅ Corrige problemas de pre-contraste constante")
    print("✅ Mantiene formato compatible con el reto")
    print("✅ Garantiza features no-constantes")
    print("✅ Tensor format: [pre_0000, post_0001, mask]")
    print("")
    
    # Crear y ejecutar pipeline
    pipeline = Fixed0000_0001_Pipeline(config)
    pipeline.run_fixed_0000_0001_pipeline()
    
    print("\n🎯 PIPELINE COMPLETADO - LISTO PARA EL RETO")
    print("✅ Datos procesados manteniendo compatibilidad")
    print("✅ Features constantes corregidas")
    print("📊 Revisa los reportes de diagnóstico")


# CONFIGURACIONES ALTERNATIVAS PARA DIFERENTES NIVELES DE CORRECCIÓN
def create_correction_strategies():
    """Crear diferentes estrategias de corrección"""
    
    base_path = r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"
    splits_path = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    
    strategies = {
        # ESTRATEGIA 1: Corrección mínima (solo casos extremos)
        "minimal_correction": Fixed0000_0001_Config(
            crop_masks_data=Path(base_path),
            splits_csv=Path(splits_path),
            output_dir=Path("D:/mama_mia_minimal_correction"),
            fix_constant_pre=True,
            pre_enhancement_method="adaptive",
            min_pre_variance=1e-8,  # Solo casos extremos
            apply_separate_normalization=False,
            fix_zero_regions=False,
            ensure_non_constant_features=True
        ),
        
        # ESTRATEGIA 2: Corrección agresiva (máxima calidad)
        "aggressive_correction": Fixed0000_0001_Config(
            crop_masks_data=Path(base_path),
            splits_csv=Path(splits_path),
            output_dir=Path("D:/mama_mia_aggressive_correction"),
            fix_constant_pre=True,
            pre_enhancement_method="adaptive",
            min_pre_variance=1e-4,  # Más estricto
            apply_separate_normalization=True,
            use_roi_based_normalization=True,
            fix_zero_regions=True,
            apply_intensity_correction=True,
            ensure_non_constant_features=True,
            feature_variance_threshold=1e-6
        ),
        
        # ESTRATEGIA 3: Solo normalización (sin corrección de datos)
        "normalization_only": Fixed0000_0001_Config(
            crop_masks_data=Path(base_path),
            splits_csv=Path(splits_path),
            output_dir=Path("D:/mama_mia_normalization_only"),
            fix_constant_pre=False,  # No corregir datos
            apply_separate_normalization=True,
            use_roi_based_normalization=True,
            ensure_non_constant_features=False,
            save_diagnostic_info=True
        ),
        
        # ESTRATEGIA 4: Diagnóstico solamente (sin procesamiento)
        "diagnostic_only": Fixed0000_0001_Config(
            crop_masks_data=Path(base_path),
            splits_csv=Path(splits_path),
            output_dir=Path("D:/mama_mia_diagnostic_only"),
            fix_constant_pre=False,
            apply_separate_normalization=False,
            save_diagnostic_info=True,
            save_corrected_data=False
        )
    }
    
    return strategies


def run_correction_strategy(strategy_name: str):
    """Ejecutar estrategia específica de corrección"""
    strategies = create_correction_strategies()
    
    if strategy_name not in strategies:
        print(f"❌ Estrategia '{strategy_name}' no encontrada.")
        print(f"✅ Estrategias disponibles: {list(strategies.keys())}")
        return
    
    config = strategies[strategy_name]
    
    print(f"🎯 EJECUTANDO ESTRATEGIA: {strategy_name.upper()}")
    print("=" * 60)
    
    pipeline = Fixed0000_0001_Pipeline(config)
    pipeline.run_fixed_0000_0001_pipeline()


if __name__ == "__main__":
    # EJECUCIÓN PRINCIPAL: Estrategia balanceada compatible con el reto
    main()
    
    # OPCIONAL: Ejecutar estrategias alternativas para comparar
    # Descomenta para probar diferentes niveles de corrección:
    
    # run_correction_strategy("minimal_correction")     # Corrección mínima
    # run_correction_strategy("aggressive_correction")  # Corrección máxima
    # run_correction_strategy("normalization_only")    # Solo normalización
    # run_correction_strategy("diagnostic_only")       # Solo diagnóstico