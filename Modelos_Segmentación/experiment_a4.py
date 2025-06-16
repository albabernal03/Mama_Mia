"""
EXPERIMENTO A4: TEMPORAL [Pre, Post1, Post2, Post3]
Secuencia temporal completa
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

print("=== EXPERIMENTO A4: TEMPORAL [Pre, Post1, Post2, Post3] ===")

base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"

output_path = Path("../Dataset105_A4_Temporal")
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)

files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))[:1506]
print(f"Encontrados {len(files_0000)} archivos para procesar")

processed = 0
for i, file in enumerate(files_0000):
    patient_id = file.name.replace("_0000.nii.gz", "")
    seg_file = seg_dir / f"{patient_id}.nii.gz"
    
    if not seg_file.exists():
        continue
        
    try:
        if (i + 1) % 100 == 0:
            print(f"Procesando A4: {patient_id} ({i+1}/{len(files_0000)})")
        
        # Buscar todas las fases disponibles
        phases = []
        for phase in ["0000", "0001", "0002", "0003", "0004"]:
            phase_file = images_dir / f"{patient_id}_{phase}.nii.gz"
            if phase_file.exists():
                phases.append((phase_file, phase))
        
        if len(phases) < 2:
            continue
        
        # Procesar y guardar cada fase
        for j, (phase_file, phase_name) in enumerate(phases):
            # Cargar y normalizar
            phase_sitk = sitk.Cast(sitk.ReadImage(str(phase_file)), sitk.sitkFloat32)
            phase_array = sitk.GetArrayFromImage(phase_sitk)
            
            # Z-score normalization
            mask = phase_array > 0
            if np.sum(mask) > 0:
                mean_val = np.mean(phase_array[mask])
                std_val = np.std(phase_array[mask])
                if std_val > 0:
                    phase_array = (phase_array - mean_val) / std_val
            
            # Guardar
            result_sitk = sitk.GetImageFromArray(phase_array.astype(np.float32))
            result_sitk.CopyInformation(phase_sitk)
            
            output_file = output_path / "imagesTr" / f"case_{processed:03d}_{phase_name}.nii.gz"
            sitk.WriteImage(result_sitk, str(output_file))
        
        # Copiar segmentación
        shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
        
        processed += 1
        
    except Exception as e:
        print(f"Error A4 {patient_id}: {e}")
        continue

# Dataset JSON
channel_names = {
    "0": "Pre_Contrast",
    "1": "Post1_Contrast",
    "2": "Post2_Contrast", 
    "3": "Post3_Contrast",
    "4": "Post4_Contrast"
}

dataset_json = {
    "channel_names": channel_names,
    "labels": {"background": 0, "tumor": 1},
    "numTraining": processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A4_Temporal_Sequence",
    "description": "Temporal sequence: Pre + Post1 + Post2 + Post3 + Post4"
}

with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"A4 COMPLETADO: {processed} casos en {output_path}")
print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 105")
print("ENTRENAR: nnUNetv2_train 105 3d_fullres 0")