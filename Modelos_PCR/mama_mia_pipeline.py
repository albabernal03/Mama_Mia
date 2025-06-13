"""

Formato correcto: [pre, post, mask] usando crops ya procesados
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CombinedConfig:
    """Configuración para usar crops existentes"""
    # Datos ya procesados
    mama_mia_data: Path  # Tu preprocessing MAMA-MIA
    crop_masks_data: Path  # Tu crop con 15% margin
    splits_csv: Path
    
    # Output optimizado
    output_dir: Path
    
    # Parámetros de optimización
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    apply_additional_normalization: bool = True
    create_difference_image: bool = True
    
    # Formato de salida
    save_3channel_tensor: bool = True  # [pre, post, mask]
    save_individual_channels: bool = True
    save_metadata: bool = True
    
    # Performance
    n_workers: int = 4
    batch_size: int = 25

class CombinedPipeline:
    """Pipeline que usa directamente tus crops existentes"""
    
    def __init__(self, config: CombinedConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Crear directorios
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Stats
        self.stats = {
            'processed': 0,
            'failed': 0,
            'missing_files': 0,
            'total': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('CombinedPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Crear directorio si no existe
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler con encoding UTF-8
            log_file = self.config.output_dir / 'combined_pipeline.log'
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def apply_percentile_clipping(self, data: np.ndarray) -> np.ndarray:
        """Aplicar clipping adicional por percentiles"""
        if not self.config.apply_additional_normalization:
            return data
        
        non_zero = data[data > 0]
        if len(non_zero) > 0:
            p_low = np.percentile(non_zero, self.config.percentile_low)
            p_high = np.percentile(non_zero, self.config.percentile_high)
            clipped = np.clip(data, p_low, p_high)
            return clipped
        return data
    
    def apply_mama_mia_normalization(self, pre_data: np.ndarray, post_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplicar normalización estilo MAMA-MIA a los crops"""
        
        # Calcular estadísticas de TODAS las fases (como MAMA-MIA)
        all_data = np.concatenate([pre_data.flatten(), post_data.flatten()])
        non_zero_data = all_data[all_data > 0]
        
        if len(non_zero_data) > 0:
            mean_all = np.mean(non_zero_data)
            std_all = np.std(non_zero_data)
            
            if std_all > 0:
                # Z-score normalization
                pre_normalized = (pre_data - mean_all) / std_all
                post_normalized = (post_data - mean_all) / std_all
                
                # Mantener valores de fondo en 0
                pre_normalized[pre_data == 0] = 0
                post_normalized[post_data == 0] = 0
                
                return pre_normalized, post_normalized
        
        return pre_data, post_data
    
    def create_optimized_outputs(self, 
                               pre_data: np.ndarray, 
                               post_data: np.ndarray,
                               seg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Crear outputs optimizados con formato [pre, post, mask]"""
        
        # Aplicar normalización MAMA-MIA
        pre_normalized, post_normalized = self.apply_mama_mia_normalization(pre_data, post_data)
        
        # Aplicar clipping adicional si está habilitado
        pre_final = self.apply_percentile_clipping(pre_normalized)
        post_final = self.apply_percentile_clipping(post_normalized)
        
        # Crear máscara binaria
        mask_binary = (seg_data > 0).astype(np.float32)
        
        outputs = {
            'pre_contrast': pre_final.astype(np.float32),
            'post_contrast': post_final.astype(np.float32),
            'segmentation': seg_data.astype(np.uint8),
            'mask_binary': mask_binary
        }
        
        # Crear imagen de diferencia si está habilitado
        if self.config.create_difference_image:
            difference = post_final - pre_final
            outputs['difference'] = difference.astype(np.float32)
        
        # Crear tensor 3-channel FORMATO CORRECTO: [pre, post, mask]
        if self.config.save_3channel_tensor:
            tensor_3ch = np.stack([
                pre_final,    # Canal 0: pre-contraste
                post_final,   # Canal 1: post-contraste  
                mask_binary   # Canal 2: máscara binaria
            ], axis=0)
            
            outputs['tensor_3channel'] = tensor_3ch.astype(np.float32)
        
        return outputs
    
    def save_processing_metadata(self, patient_id: str, metadata: Dict) -> None:
        """Guardar metadatos de procesamiento"""
        if not self.config.save_metadata:
            return
            
        metadata_file = self.config.output_dir / patient_id / "processing_metadata.json"
        metadata_file.parent.mkdir(exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            # Convertir numpy types a tipos serializables
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                else:
                    return obj
            
            serializable_metadata = convert_numpy(metadata)
            json.dump(serializable_metadata, f, indent=2)
    
    def process_patient(self, patient_id: str, split_type: str) -> bool:
        """Procesar paciente usando directamente crops existentes"""
        
        try:
            # 1. BUSCAR CROPS EXISTENTES
            crop_img_dir = self.config.crop_masks_data / "images" / split_type / patient_id
            crop_seg_dir = self.config.crop_masks_data / "segmentations" / split_type / patient_id
            
            # Verificar que existen los directorios
            if not crop_img_dir.exists() or not crop_seg_dir.exists():
                self.logger.warning(f"Crop directories not found for {patient_id}")
                self.stats['missing_files'] += 1
                return False
            
            # 2. BUSCAR ARCHIVOS CROP
            crop_img_files = list(crop_img_dir.glob("*_cropped.nii.gz"))
            crop_seg_files = list(crop_seg_dir.glob("*_cropped.nii.gz"))
            
            if not crop_img_files or not crop_seg_files:
                self.logger.warning(f"Crop files not found for {patient_id}")
                self.stats['missing_files'] += 1
                return False
            
            # 3. IDENTIFICAR ARCHIVOS PRE Y POST
            pre_crop_file = None
            post_crop_file = None
            
            # Buscar por patrones en nombres
            for crop_file in crop_img_files:
                filename = crop_file.name.lower()
                if "0000" in filename or "pre" in filename:
                    pre_crop_file = crop_file
                elif "0001" in filename or "post" in filename:
                    post_crop_file = crop_file
                elif "_cropped" in filename and not pre_crop_file:
                    # Si no hay patrones claros, tomar el primero como post
                    post_crop_file = crop_file
            
            seg_crop_file = crop_seg_files[0]
            
            # Verificar archivos esenciales
            if not post_crop_file or not seg_crop_file:
                self.logger.warning(f"Required crop files not found for {patient_id}")
                self.stats['missing_files'] += 1
                return False
            
            # 4. CARGAR DATOS CROP
            post_data = nib.load(post_crop_file).get_fdata()
            seg_data = nib.load(seg_crop_file).get_fdata()
            
            # Pre-contrast (usar si existe, sino simular)
            if pre_crop_file and pre_crop_file.exists():
                pre_data = nib.load(pre_crop_file).get_fdata()
            else:
                # Simular pre-contrast
                pre_data = post_data * 0.6
                self.logger.info(f"Using simulated pre-contrast for {patient_id}")
            
            # 5. VERIFICAR QUE LOS DATOS SON VÁLIDOS
            if post_data.shape != seg_data.shape:
                self.logger.warning(f"Shape mismatch for {patient_id}: post={post_data.shape}, seg={seg_data.shape}")
                return False
            
            if np.sum(seg_data > 0) == 0:
                self.logger.warning(f"Empty segmentation for {patient_id}")
                return False
            
            # 6. CREAR OUTPUTS OPTIMIZADOS
            outputs = self.create_optimized_outputs(pre_data, post_data, seg_data)
            
            # 7. GUARDAR RESULTADOS
            output_patient_dir = self.config.output_dir / patient_id
            output_patient_dir.mkdir(exist_ok=True)
            
            # Tensor principal FORMATO CORRECTO: [pre, post, mask]
            if self.config.save_3channel_tensor and 'tensor_3channel' in outputs:
                tensor_img = nib.Nifti1Image(outputs['tensor_3channel'], np.eye(4))
                nib.save(tensor_img, output_patient_dir / f"{patient_id}_tensor_3ch.nii.gz")
            
            # Canales individuales si está habilitado
            if self.config.save_individual_channels:
                for key, data in outputs.items():
                    if key != 'tensor_3channel':
                        output_img = nib.Nifti1Image(data, np.eye(4))
                        nib.save(output_img, output_patient_dir / f"{patient_id}_{key}.nii.gz")
            
            # 8. GUARDAR METADATOS
            if self.config.save_metadata:
                metadata = {
                    'patient_id': patient_id,
                    'split_type': split_type,
                    'cropped_shape': list(post_data.shape),
                    'crop_sources': {
                        'pre_crop': str(pre_crop_file) if pre_crop_file else 'simulated',
                        'post_crop': str(post_crop_file),
                        'seg_crop': str(seg_crop_file)
                    },
                    'outputs_created': list(outputs.keys()),
                    'tensor_format': '[pre, post, mask]',
                    'processing_method': 'direct_crop_usage',
                    'config': {
                        'percentile_clipping': self.config.apply_additional_normalization,
                        'percentile_range': [self.config.percentile_low, self.config.percentile_high],
                        'difference_image': self.config.create_difference_image,
                        'tensor_3channel': self.config.save_3channel_tensor,
                        'tensor_channels': 'pre_contrast, post_contrast, mask_binary'
                    },
                    'stats': {
                        'tumor_volume_voxels': int(np.sum(seg_data > 0)),
                        'pre_mean': float(np.mean(pre_data[pre_data > 0])) if np.any(pre_data > 0) else 0,
                        'post_mean': float(np.mean(post_data[post_data > 0])) if np.any(post_data > 0) else 0,
                        'pre_std': float(np.std(pre_data[pre_data > 0])) if np.any(pre_data > 0) else 0,
                        'post_std': float(np.std(post_data[post_data > 0])) if np.any(post_data > 0) else 0,
                        'mask_coverage': float(np.sum(seg_data > 0) / np.prod(seg_data.shape))
                    }
                }
                
                self.save_processing_metadata(patient_id, metadata)
            
            self.stats['processed'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {patient_id}: {str(e)}")
            self.stats['failed'] += 1
            return False
    
    def create_summary_report(self) -> None:
        """Crear reporte resumen del procesamiento"""
        
        summary_stats = {
            'processing_summary': {
                'total_patients': self.stats['total'],
                'successfully_processed': self.stats['processed'],
                'failed': self.stats['failed'],
                'missing_files': self.stats['missing_files'],
                'success_rate': f"{self.stats['processed']/self.stats['total']*100:.1f}%" if self.stats['total'] > 0 else "0%"
            },
            'tensor_format': {
                'channels': '[pre, post, mask]',
                'description': 'Channel 0: pre-contrast, Channel 1: post-contrast, Channel 2: binary mask',
                'processing_method': 'direct_crop_usage_with_mama_mia_normalization'
            },
            'configuration': {
                'percentile_clipping': self.config.apply_additional_normalization,
                'percentile_range': [self.config.percentile_low, self.config.percentile_high],
                'difference_image_created': self.config.create_difference_image,
                'tensor_3channel_created': self.config.save_3channel_tensor,
                'individual_channels_saved': self.config.save_individual_channels
            },
            'data_sources': {
                'crop_masks_data': str(self.config.crop_masks_data),
                'output_directory': str(self.config.output_dir)
            }
        }
        
        # Guardar reporte
        report_file = self.config.output_dir / "processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        self.logger.info(f"Report saved: {report_file}")
    
    def run_combined_pipeline(self) -> None:
        """Ejecutar pipeline usando crops existentes"""
        
        self.logger.info("PROCESSING USING EXISTING CROPS - [pre, post, mask] FORMAT")
        self.logger.info("=" * 65)
        self.logger.info(f"Sources:")
        self.logger.info(f"   Crop data: {self.config.crop_masks_data}")
        self.logger.info(f"Output: {self.config.output_dir}")
        self.logger.info(f"Tensor format: [pre, post, mask] using direct crops")
        self.logger.info(f"Configuration:")
        self.logger.info(f"   MAMA-MIA normalization: Applied to crops")
        self.logger.info(f"   Percentile clipping: {self.config.apply_additional_normalization}")
        self.logger.info(f"   3-channel tensor: {self.config.save_3channel_tensor}")
        self.logger.info("")
        
        # Cargar splits
        df = pd.read_csv(self.config.splits_csv)
        
        all_patients = []
        for split_type in ['train_split', 'test_split']:
            patients = df[split_type].dropna().tolist()
            for patient_id in patients:
                all_patients.append((patient_id, split_type))
        
        self.stats['total'] = len(all_patients)
        self.logger.info(f"Total patients: {self.stats['total']}")
        
        # Procesamiento
        if self.config.n_workers > 1:
            # Paralelo
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                future_to_patient = {
                    executor.submit(self.process_patient, patient_id, split_type): (patient_id, split_type)
                    for patient_id, split_type in all_patients
                }
                
                for future in as_completed(future_to_patient):
                    patient_id, split_type = future_to_patient[future]
                    try:
                        future.result()
                        if (self.stats['processed'] + self.stats['failed']) % self.config.batch_size == 0:
                            self.logger.info(f"    Progress: {self.stats['processed']}/{self.stats['total']}")
                    except Exception as e:
                        self.logger.error(f"Error in {patient_id}: {e}")
        else:
            # Secuencial
            for i, (patient_id, split_type) in enumerate(all_patients):
                self.process_patient(patient_id, split_type)
                if (i + 1) % self.config.batch_size == 0:
                    self.logger.info(f"    Progress: {self.stats['processed']}/{self.stats['total']}")
        
        # Crear reporte final
        self.create_summary_report()
        
        # Reporte final
        self.logger.info("\n" + "=" * 65)
        self.logger.info("PROCESSING COMPLETED - [pre, post, mask] FORMAT")
        self.logger.info(f"Results:")
        self.logger.info(f"   Successfully processed: {self.stats['processed']}")
        self.logger.info(f"   Failed: {self.stats['failed']}")
        self.logger.info(f"   Missing files: {self.stats['missing_files']}")
        self.logger.info(f"   Success rate: {self.stats['processed']/self.stats['total']*100:.1f}%")
        self.logger.info(f"\nDATA READY IN: {self.config.output_dir}")
        self.logger.info(f"Tensor format: [pre, post, mask] - Ready for training!")


def main():
    """Ejecutar pipeline usando crops existentes con formato [pre, post, mask]"""
    config = CombinedConfig(
        mama_mia_data=Path(r"D:\preprocessed_mama_mia_style"),  # Solo para referencia
        crop_masks_data=Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"),
        splits_csv=Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"),
        output_dir=Path("D:/mama_mia_final_corrected"),  # NUEVO DIRECTORIO
        
        apply_additional_normalization=True,
        create_difference_image=True,
        save_3channel_tensor=True,      # [pre, post, mask]
        save_individual_channels=True,
        save_metadata=True,
        
        n_workers=4,
        batch_size=25
    )
    
    # Crear y ejecutar pipeline
    pipeline = CombinedPipeline(config)
    pipeline.run_combined_pipeline()
    
    print("\nPIPELINE COMPLETADO - USANDO CROPS DIRECTAMENTE")
    print("Tensor format: [pre, post, mask]")
    print("Ready for model training!")

if __name__ == "__main__":
    main()