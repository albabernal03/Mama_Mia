#!/usr/bin/env python3
"""
MAMA-MIA Challenge - Modelo Base con Preprocesamiento Óptimo
Basado en Schwarzhans et al. (2025): Bias Field + Spatial + Z-score normalization

Autor: Alba 
Fecha: Mayo 2025
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import shutil
import json
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")

class MAMAMIAPreprocessor:
    def __init__(self, base_dir: str, output_dir: str = "nnUNet_preprocessed"):
        """
        Inicializa el preprocesador MAMA-MIA
        
        Args:
            base_dir: Directorio base con carpetas 'images' y 'segmentations'
            output_dir: Directorio de salida para nnU-Net
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.base_dir / "images"
        self.segmentations_dir = self.base_dir / "segmentations"
        
        # Crear estructura nnU-Net
        self.setup_nnunet_structure()
        
    def setup_nnunet_structure(self):
        """Crea la estructura de directorios para nnU-Net"""
        self.nnunet_dir = self.output_dir / "Dataset001_MAMA_MIA"
        
        # Directorios principales
        (self.nnunet_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "labelsTs").mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Estructura nnU-Net creada en: {self.nnunet_dir}")
    
    def n4_bias_correction(self, image_sitk: sitk.Image) -> sitk.Image:
        """
        Aplica corrección de campo de sesgo N4ITK
        
        Args:
            image_sitk: Imagen SimpleITK
            
        Returns:
            Imagen corregida
        """
        # Crear máscara de tejido (voxeles > 0)
        mask = sitk.BinaryThreshold(image_sitk, lowerThreshold=1)
        
        # Aplicar N4 Bias Field Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaskImage(mask)
        corrected_image = corrector.Execute(image_sitk)
        
        return corrected_image
    
    def spatial_normalization(self, image_sitk: sitk.Image, 
                            target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                            target_size: Optional[Tuple[int, int, int]] = None) -> sitk.Image:
        """
        Normalización espacial: resampleo a voxel isotrópico
        
        Args:
            image_sitk: Imagen SimpleITK
            target_spacing: Spacing objetivo (1x1x1 mm por defecto)
            target_size: Tamaño objetivo (opcional)
            
        Returns:
            Imagen resampleada
        """
        original_spacing = image_sitk.GetSpacing()
        original_size = image_sitk.GetSize()
        
        # Calcular nuevo tamaño si no se especifica
        if target_size is None:
            target_size = [
                int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
        
        # Configurar resampleo
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)  # Interpolación suave
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        
        # Resamplear
        resampled_image = resampler.Execute(image_sitk)
        
        return resampled_image
    
    def zscore_normalization(self, image_array: np.ndarray, 
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score normalization optimizada según Schwarzhans et al.
        
        Args:
            image_array: Array de la imagen
            mask: Máscara para calcular estadísticas (opcional)
            
        Returns:
            Imagen normalizada
        """
        if mask is None:
            # Crear máscara excluyendo fondo (voxeles = 0)
            mask = image_array > 0
        
        if np.sum(mask) == 0:
            print("⚠️  Warning: Máscara vacía, usando toda la imagen")
            mask = np.ones_like(image_array, dtype=bool)
        
        # Calcular estadísticas solo en la región de interés
        mean_val = np.mean(image_array[mask])
        std_val = np.std(image_array[mask])
        
        # Evitar división por cero
        if std_val == 0:
            std_val = 1.0
            print("⚠️  Warning: Desviación estándar = 0, usando std = 1")
        
        # Aplicar z-score
        normalized_image = (image_array - mean_val) / std_val
        
        return normalized_image
    
    def create_subtraction_image(self, pre_path: str, post_path: str) -> np.ndarray:
        """
        Crea imagen de resta (Post1 - Pre) optimizada
        
        Args:
            pre_path: Ruta imagen pre-contraste
            post_path: Ruta imagen post-contraste
            
        Returns:
            Imagen de resta como array numpy
        """
        # Cargar imágenes
        pre_sitk = sitk.ReadImage(str(pre_path))
        post_sitk = sitk.ReadImage(str(post_path))
        
        # Aplicar bias field correction a ambas
        print("  📐 Aplicando N4 Bias Field Correction...")
        pre_corrected = self.n4_bias_correction(pre_sitk)
        post_corrected = self.n4_bias_correction(post_sitk)
        
        # Normalización espacial
        print("  📏 Normalizando espacialmente...")
        pre_resampled = self.spatial_normalization(pre_corrected)
        post_resampled = self.spatial_normalization(post_corrected)
        
        # Convertir a numpy
        pre_array = sitk.GetArrayFromImage(pre_resampled)
        post_array = sitk.GetArrayFromImage(post_resampled)
        
        # Crear imagen de resta
        subtraction_array = post_array - pre_array
        
        # Z-score normalization
        print("  📊 Aplicando Z-score normalization...")
        subtraction_normalized = self.zscore_normalization(subtraction_array)
        
        # Convertir de vuelta a SimpleITK para mantener metadatos
        subtraction_sitk = sitk.GetImageFromArray(subtraction_normalized)
        subtraction_sitk.CopyInformation(post_resampled)
        
        return subtraction_sitk
    
    def get_patient_files(self) -> List[str]:
        """
        Obtiene lista de pacientes disponibles
        
        Returns:
            Lista de IDs de pacientes
        """
        patients = set()
        
        # Buscar archivos _0000 y _0001
        for file in self.images_dir.glob("*_0000.nii.gz"):
            patient_id = file.stem.replace("_0000", "")
            post_file = self.images_dir / f"{patient_id}_0001.nii.gz"
            
            if post_file.exists():
                patients.add(patient_id)
        
        return sorted(list(patients))
    
    def process_single_patient(self, patient_id: str) -> bool:
        """
        Procesa un paciente individual
        
        Args:
            patient_id: ID del paciente
            
        Returns:
            True si se procesó correctamente
        """
        try:
            # Rutas de archivos
            pre_path = self.images_dir / f"{patient_id}_0000.nii.gz"
            post_path = self.images_dir / f"{patient_id}_0001.nii.gz"
            
            if not pre_path.exists() or not post_path.exists():
                print(f"❌ Archivos faltantes para {patient_id}")
                return False
            
            print(f"🔄 Procesando {patient_id}...")
            
            # Crear imagen de resta con preprocesamiento completo
            subtraction_sitk = self.create_subtraction_image(pre_path, post_path)
            
            # Guardar imagen procesada
            output_path = self.nnunet_dir / "imagesTr" / f"{patient_id}_0000.nii.gz"
            sitk.WriteImage(subtraction_sitk, str(output_path))
            
            # Copiar segmentación experta si existe
            seg_path = self.segmentations_dir / "expert" / f"{patient_id}.nii.gz"
            if seg_path.exists():
                output_seg_path = self.nnunet_dir / "labelsTr" / f"{patient_id}.nii.gz"
                shutil.copy2(seg_path, output_seg_path)
            else:
                print(f"⚠️  Segmentación experta no encontrada para {patient_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {str(e)}")
            return False
    
    def create_dataset_json(self, patient_list: List[str]):
        """
        Crea el archivo dataset.json para nnU-Net
        
        Args:
            patient_list: Lista de pacientes procesados
        """
        dataset_json = {
            "channel_names": {
                "0": "Subtraction_DCE_MRI"
            },
            "labels": {
                "background": 0,
                "tumor": 1
            },
            "numTraining": len(patient_list),
            "file_ending": ".nii.gz",
            "dataset_name": "MAMA_MIA_Challenge",
            "description": "MAMA-MIA breast cancer DCE-MRI segmentation with optimal preprocessing",
            "reference": "Schwarzhans et al. (2025) - Bias Field + Spatial + Z-score normalization",
            "release": "1.0"
        }
        
        # Guardar JSON
        json_path = self.nnunet_dir / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"✅ Dataset JSON creado: {json_path}")
    
    def run_preprocessing(self, train_split: float = 0.8):
        """
        Ejecuta todo el preprocesamiento
        
        Args:
            train_split: Proporción para entrenamiento (0.8 = 80%)
        """
        print("🚀 Iniciando preprocesamiento MAMA-MIA...")
        print("📋 Configuración:")
        print("   - Bias Field Correction: N4ITK")
        print("   - Spatial Normalization: 1x1x1 mm")
        print("   - Intensity Normalization: Z-score")
        print("   - Input: Subtraction Image (Post1 - Pre)")
        print()
        
        # Obtener lista de pacientes
        patients = self.get_patient_files()
        print(f"📊 Pacientes encontrados: {len(patients)}")
        
        if len(patients) == 0:
            print("❌ No se encontraron pacientes válidos")
            return
        
        # Procesar pacientes
        processed_patients = []
        for patient_id in tqdm(patients, desc="Procesando pacientes"):
            if self.process_single_patient(patient_id):
                processed_patients.append(patient_id)
        
        print(f"\n✅ Pacientes procesados exitosamente: {len(processed_patients)}/{len(patients)}")
        
        # Crear dataset.json
        self.create_dataset_json(processed_patients)
        
        # Estadísticas finales
        print(f"\n📈 RESUMEN:")
        print(f"   - Directorio de salida: {self.nnunet_dir}")
        print(f"   - Imágenes de entrenamiento: {len(processed_patients)}")
        print(f"   - Preprocesamiento: Bias Field + Spatial + Z-score")
        print(f"   - Listo para nnU-Net v2!")
        
        # Comando para nnU-Net
        print(f"\n🔥 PRÓXIMO PASO - Entrenar con nnU-Net:")
        print(f"   export nnUNet_raw='{self.output_dir}'")
        print(f"   export nnUNet_preprocessed='{self.output_dir}/nnUNet_preprocessed'")
        print(f"   export nnUNet_results='{self.output_dir}/nnUNet_results'")
        print(f"   nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
        print(f"   nnUNetv2_train 001 3d_fullres 0")


def main():
    parser = argparse.ArgumentParser(description="MAMA-MIA Preprocesamiento Óptimo")
    parser.add_argument("--base_dir", type=str, required=True, 
                       help="Directorio base con carpetas 'images' y 'segmentations'")
    parser.add_argument("--output_dir", type=str, default="nnUNet_preprocessed",
                       help="Directorio de salida")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Proporción para entrenamiento")
    
    args = parser.parse_args()
    
    # Crear y ejecutar preprocesador
    preprocessor = MAMAMIAPreprocessor(args.base_dir, args.output_dir)
    preprocessor.run_preprocessing(args.train_split)


if __name__ == "__main__":
    main()