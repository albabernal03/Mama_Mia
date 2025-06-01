"""
EXPERIMENTO A1: BASELINE - Subtraction Only (Post1 - Pre)
Solo Z-score normalization, imagen de resta
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

print("=== EXPERIMENTO A1: BASELINE SUBTRACTION ===")

base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"

output_path = Path("../Dataset101_A1_Baseline")
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)

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
        if (i + 1) % 100 == 0:
            print(f"Procesando A1: {patient_id} ({i+1}/{len(files_0000)})")
        
        # Cargar imágenes
        pre_sitk = sitk.Cast(sitk.ReadImage(str(file)), sitk.sitkFloat32)
        post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)
        
        # Crear resta
        pre_array = sitk.GetArrayFromImage(pre_sitk)
        post_array = sitk.GetArrayFromImage(post_sitk)
        sub_array = post_array - pre_array
        
        # Z-score normalization
        mask = sub_array > 0
        if np.sum(mask) > 0:
            mean_val = np.mean(sub_array[mask])
            std_val = np.std(sub_array[mask])
            if std_val > 0:
                sub_array = (sub_array - mean_val) / std_val
        
        # Guardar
        result_sitk = sitk.GetImageFromArray(sub_array.astype(np.float32))
        result_sitk.CopyInformation(post_sitk)
        
        sitk.WriteImage(result_sitk, str(output_path / "imagesTr" / f"case_{processed:03d}_0000.nii.gz"))
        shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
        
        processed += 1
        
    except Exception as e:
        print(f"Error A1 {patient_id}: {e}")
        continue

# Dataset JSON
dataset_json = {
    "channel_names": {"0": "Subtraction_Baseline"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A1_Baseline_Subtraction",
    "description": "Baseline: Post1-Pre with Z-score normalization only"
}

with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"A1 COMPLETADO: {processed} casos en {output_path}")
print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 101")
print("ENTRENAR: nnUNetv2_train 101 3d_fullres 0")