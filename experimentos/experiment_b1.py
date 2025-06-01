"""
EXPERIMENTO B1: PREPROCESAMIENTO ÓPTIMO
N4 Bias Field + Spatial Normalization + Z-score
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

print("=== EXPERIMENTO B1: PREPROCESAMIENTO ÓPTIMO ===")

base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"

output_path = Path("../Dataset104_B1_Optimal")
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)

def apply_n4(image_sitk):
    """Aplicar N4 Bias Field Correction"""
    try:
        image_float = sitk.Cast(image_sitk, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)
        return corrector.Execute(image_float)
    except:
        return sitk.Cast(image_sitk, sitk.sitkFloat32)

def resample_to_1mm(image_sitk):
    """Resamplear a 1x1x1 mm"""
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()
    
    target_spacing = (1.0, 1.0, 1.0)
    target_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    
    return resampler.Execute(image_sitk)

files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))[:1506]
print(f"Encontrados {len(files_0000)} archivos para procesar")

processed = 0
for i, file in enumerate(files_0000):
    patient_id = file.name.replace("_0000.nii.gz", "")
    post_file = images_dir / f"{patient_id}_0001.nii.gz"
    seg_file = seg_dir / f"{patient_id}.nii.gz"
    
    if not post_file.exists() or not seg_file.exists():
        continue
        
    try:
        if (i + 1) % 50 == 0:
            print(f"Procesando B1: {patient_id} ({i+1}/{len(files_0000)})")
        
        # Cargar imágenes
        pre_sitk = sitk.ReadImage(str(file))
        post_sitk = sitk.ReadImage(str(post_file))
        
        # Paso 1: N4 Bias Field Correction
        pre_n4 = apply_n4(pre_sitk)
        post_n4 = apply_n4(post_sitk)
        
        # Paso 2: Spatial Normalization
        pre_resampled = resample_to_1mm(pre_n4)
        post_resampled = resample_to_1mm(post_n4)
        
        # Paso 3: Crear resta
        pre_array = sitk.GetArrayFromImage(pre_resampled)
        post_array = sitk.GetArrayFromImage(post_resampled)
        sub_array = post_array - pre_array
        
        # Paso 4: Z-score normalization
        mask = sub_array > 0
        if np.sum(mask) > 0:
            mean_val = np.mean(sub_array[mask])
            std_val = np.std(sub_array[mask])
            if std_val > 0:
                sub_array = (sub_array - mean_val) / std_val
        
        # Guardar
        result_sitk = sitk.GetImageFromArray(sub_array.astype(np.float32))
        result_sitk.CopyInformation(post_resampled)
        
        sitk.WriteImage(result_sitk, str(output_path / "imagesTr" / f"case_{processed:03d}_0000.nii.gz"))
        shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
        
        processed += 1
        
    except Exception as e:
        print(f"Error B1 {patient_id}: {e}")
        continue

# Dataset JSON
dataset_json = {
    "channel_names": {"0": "Optimal_Subtraction"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": processed,
    "file_ending": ".nii.gz",
    "dataset_name": "B1_Optimal_Preprocessing",
    "description": "Optimal: N4 + Spatial(1mm) + Z-score normalization"
}

with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"B1 COMPLETADO: {processed} casos en {output_path}")
print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 104")
print("ENTRENAR: nnUNetv2_train 104 3d_fullres 0")