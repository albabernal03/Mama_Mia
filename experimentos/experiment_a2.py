#!/usr/bin/env python3
"""
EXPERIMENTO A2: POST-CONTRASTE ONLY
Solo imagen post-contraste, sin resta
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

def create_experiment_A2(base_dir, output_dir="Dataset102_A2_PostOnly", num_cases=None):
    """
    EXPERIMENTO A2: Solo Post-contraste
    """
    print("=== EXPERIMENTO A2: POST-CONTRASTE ONLY ===")
    
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    seg_dir = base_path / "segmentations" / "expert"
    
    # Crear estructura nnU-Net
    output_path = Path(output_dir)
    (output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    # Obtener pacientes
    files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))
    if num_cases:
        files_0000 = files_0000[:num_cases]
    
    processed = 0
    for file in files_0000:
        patient_id = file.name.replace("_0000.nii.gz", "")
        post_file = images_dir / f"{patient_id}_0001.nii.gz"
        seg_file = seg_dir / f"{patient_id}.nii.gz"
        
        if not post_file.exists() or not seg_file.exists():
            continue
            
        try:
            print(f"Procesando A2: {patient_id} ({processed+1})")
            
            # Cargar solo post-contraste
            post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)
            post_array = sitk.GetArrayFromImage(post_sitk)
            
            # Z-score normalization
            mask = post_array > 0
            if np.sum(mask) > 0:
                mean_val = np.mean(post_array[mask])
                std_val = np.std(post_array[mask])
                if std_val > 0:
                    post_array = (post_array - mean_val) / std_val
            
            # Guardar
            result_sitk = sitk.GetImageFromArray(post_array.astype(np.float32))
            result_sitk.CopyInformation(post_sitk)
            
            sitk.WriteImage(result_sitk, str(output_path / "imagesTr" / f"case_{processed:03d}_0000.nii.gz"))
            shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
            
            processed += 1
            
        except Exception as e:
            print(f"Error A2 {patient_id}: {e}")
            continue
    
    # Dataset JSON
    dataset_json = {
        "channel_names": {"0": "Post_Contrast_Only"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": processed,
        "file_ending": ".nii.gz",
        "dataset_name": "A2_Post_Contrast_Only",
        "description": "Post-contrast image only with Z-score normalization"
    }
    
    with open(output_path / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"A2 COMPLETADO: {processed} casos en {output_path}")
    return processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--num_cases", type=int, default=None)
    args = parser.parse_args()
    
    create_experiment_A2(args.base_dir, num_cases=args.num_cases)
    print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 102")
    print("ENTRENAR: nnUNetv2_train 102 3d_fullres 0")